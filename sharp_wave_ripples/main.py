#!/usr/bin/env python

import sys
import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import pylab as plt
import neuron
import LFPy
from neuron import h


root_folder = '..'
layer_thickness = 200
random_seed = 123
random_seed_shift = {
                     "hbp_L4_SS_cADpyr230_1": 5,
                     'hbp_L5_TTPC2_cADpyr232_1': 6,
                     "hbp_L4_SBC_bNAC219_1": 7,}

cell_type_weight_scale = {"hbp_L4_SS_cADpyr230_1": 0.1,
                          'hbp_L5_TTPC2_cADpyr232_1': 1.0,
                          "hbp_L4_SBC_bNAC219_1": 0.0075,}


def initialize_population(num_cells, celltype):
    print "Initializing cell positions and rotations ..."
    cell_density = 100000. * 1e-9 #  cells / um^3

    pop_radius = np.sqrt(num_cells / (cell_density * np.pi * layer_thickness))
    print num_cells, pop_radius

    x_y_z_rot = np.zeros((num_cells, 4))

    for cell_number in range(num_cells):
        plt.seed((random_seed + random_seed_shift[celltype]) * cell_number + 1)
        rotation = 2*np.pi*np.random.random()

        z = layer_thickness * (np.random.random() - 0.5)
        x, y = 2 * (np.random.random(2) - 0.5) * pop_radius
        while np.sqrt(x**2 + y**2) > pop_radius:
            x, y = 2 * (np.random.random(2) - 0.5) * pop_radius
        x_y_z_rot[cell_number, :] = [x, y, z, rotation]

    r = np.array([np.sqrt(d[0]**2 + d[1]**2) for d in x_y_z_rot])
    argsort = np.argsort(r)
    x_y_z_rot = x_y_z_rot[argsort]

    np.save(join(root_folder, 'x_y_z_rot_%d_%s.npy' % (num_cells, celltype)), x_y_z_rot)
    plt.close("all")
    plt.subplot(121, xlabel='x', ylabel='y', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 1], edgecolor='none', s=2)

    plt.subplot(122, xlabel='x', ylabel='z', aspect=1, frameon=False)
    plt.scatter(x_y_z_rot[:, 0], x_y_z_rot[:, 2], edgecolor='none', s=2)

    plt.savefig(join(root_folder, 'population_%d_%s.png' % (num_cells, celltype)))


def make_synapse(cell, weight, input_idx, input_spike_train, e=0.):

    synapse_parameters = {
        'idx': input_idx,
        'e': e,
        'syntype': 'Exp2Syn',
        # 'tau': 2.,
        'tau1' : 1.,                #Time constant, rise
        'tau2' : 3.,                #Time constant, decay
        # 'tau2': 2,
        'weight': weight,
        'record_current': False,
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(input_spike_train)
    return cell, synapse

def return_cell(celltype, conductance_type="passive", cell_number=None):
    ### MAKING THE CELL

    if "hbp" in celltype:
        model_folder = join("/home", "tone", "work", "hbp_cell_models")
        # mod_folder = join(model_folder, "mods")
        # sys.path.append(model_folder)
        # import hbp_cell_models
        from hbp_cell_models import return_cell as hbp_return_cell
        from hbp_cell_models import remove_mechanisms
        from hbp_cell_models import make_cell_uniform

        remove_lists = {'active': [],
                        'passive': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                                    "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                                    "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"],
                        'Ih': ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                               "SK_E2", "K_Tst", "K_Pst", "Im",
                               "CaDynamics_E2", "Ca_LVAst", "Ca"]}

    if "hbp" in celltype:

        v_rest = -75.
        cell_name = celltype.replace("hbp_", "")
        cell_folder = join(model_folder, "models", cell_name)
        end_T = 150
        dt = 2**-5
        start_T = -200
        cell = hbp_return_cell(cell_folder, end_T, dt, start_T, v_init=v_rest)

        remove_mechanisms(remove_lists[conductance_type])
        # make_cell_uniform(v_rest)
        # cell = fix_cell_diam(cell, cell_parameters)
        # cell.set_rotation(x=np.pi/2, y=0.0)
        cell.set_rotation(x=0)

    else:
        raise RuntimeError("No cell")

    if cell_number is not None:
        cell_x_y_z_rotation = np.load(join(root_folder, 'x_y_z_rot_%d_%s.npy' % (10000, celltype)))
        cell.set_rotation(z=cell_x_y_z_rotation[cell_number][3])
        z_shift = np.max(cell.zend) + layer_thickness / 2
        cell.set_pos(x=cell_x_y_z_rotation[cell_number][0],
                     y=cell_x_y_z_rotation[cell_number][1],
                     z=cell_x_y_z_rotation[cell_number][2] - z_shift)
        if np.max(cell.zend) > 0:
            raise RuntimeError("Cell reaches above cortex")

    return cell

def animate_vmem(cell, fig_name):

    vmin = -70
    vmax = -60
    print vmin, vmax

    dz = 50.
    z_box, vsd = return_z_resolved_vsd(cell, dz)

    palette = plt.cm.hot
    palette.set_bad('gray', 1.0) # Bad values (i.e., masked, set to grey 0.8
    p_dict = {'cmap': palette, "origin": "lower", "interpolation": "nearest", "vmax": vmax, "vmin": vmin,
              'extent':[-200, 200, np.min(z_box), np.max(z_box)], "zorder": 0}

    VSD = np.ma.array(vsd, mask=np.isnan(vsd))

    if not os.path.isdir(join(root_folder, "vmem_anim")):
        os.mkdir(join(root_folder, "vmem_anim"))

    cmaps = [plt.cm.hot]
    vmem_clr = lambda vmem, cmap_func: cmap_func(0.0 + 1.0 * (vmem - vmin) / (vmax - vmin))

    plt.close('all')

    fig = plt.figure(figsize=[6, 6])
    fig.subplots_adjust(wspace=0.6, hspace=0.6)

    ax_1 = fig.add_subplot(121, aspect=1, frameon=False, ylim=[-900, 100])#, xticks=[], yticks=[])
    ax_vsd = fig.add_subplot(122, aspect=1, frameon=False, ylim=[-900, 100], sharey=ax_1)#, xticks=[], yticks=[])

    img = ax_1.imshow([[]], vmin=vmin, vmax=vmax, origin='lower', cmap=cmaps[0])
    plt.colorbar(img, ax=ax_1, shrink=0.5)

    name = fig.suptitle("t = %1.1f ms" % 0.0)
    img3 = ax_vsd.imshow(VSD[:, 0].reshape(1,-1).T, **p_dict)

    plt.colorbar(img3, ax=ax_vsd, label='VSD', shrink=0.5)

    morph_line_dict = {'solid_capstyle': 'butt',
                       'lw': 0.5,
                       'color': 'k',
                       'zorder': 1}
    lines = []
    lines2 = []
    num_tsteps = len(cell.tvec)
    t_idx = 0

    cmap = np.random.choice(cmaps)
    # cell_cmaps.append(cmap)

    for idx in xrange(len(cell.xend)):
        l = None
        if idx == 0:
            l, = ax_1.plot(cell.xmid[idx], cell.zmid[idx],
                    'o', ms=8, zorder=0, mec='none',
                     c=vmem_clr(cell.vmem[idx, t_idx], cmap))
        else:
            x = [cell.xstart[idx], cell.xend[idx]]
            z = [cell.zstart[idx], cell.zend[idx]]
            l, = ax_1.plot(x, z, c=vmem_clr(cell.vmem[idx, t_idx], cmap),
                    lw=2, zorder=0, clip_on=False)

        l2, =ax_vsd.plot(cell.xmid[idx], cell.zmid[idx],
                'o', ms=4,  zorder=2, #mec='none',
                 c=vmem_clr(cell.vmem[idx, t_idx], cmap))

        lines2.append(l2)
        lines.append(l)

    for t_idx in range(num_tsteps)[::5][1:]:
        print cell.tvec[t_idx], t_idx, num_tsteps
        name.set_text("t = %1.1f ms" % cell.tvec[t_idx])
        for idx in xrange(len(cell.xend)):
            lines[idx].set_color(vmem_clr(cell.vmem[idx, t_idx], cmap))
            lines2[idx].set_color(vmem_clr(cell.vmem[idx, t_idx], cmap))
        plt.draw()
        img3 = ax_vsd.imshow(VSD[:, t_idx].reshape(1,-1).T, **p_dict)

        plt.savefig(join(root_folder, "vmem_anim", 'anim_%s_%04d.png' % (fig_name, t_idx)))


def make_input(cell, input_type, cell_type):

    weight = 0.002 * cell_type_weight_scale[cell_type]
    if input_type == "bombardment":
        input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=1000)
        for input_idx in input_idxs:
            input_spike_train = np.array([cell.tstartms + (cell.tstopms - cell.tstartms) * np.random.random()])
            cell, synapse = make_synapse(cell, weight * np.random.normal(1., 0.2), input_idx, input_spike_train)

    elif input_type == "waves":

        # input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=100)
        # for input_idx in input_idxs:
        #     input_spike_train = np.array([cell.tstartms + (cell.tstopms - cell.tstartms) * np.random.random()])
        #     cell, synapse = make_synapse(cell, weight * np.random.normal(1., 0.2), input_idx, input_spike_train)

        # Waves arriving in the soma region
        input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=100)
                                                 # z_min=cell.zmid[0]-100,
                                                 # z_max=cell.zmid[0]+100)
        # Input is centered around:
        input_center = 30
        input_std = 0.25
        for input_idx in input_idxs:
            input_spike_train = np.random.normal(input_center, input_std,
                                                 size=1)
            cell, synapse = make_synapse(cell,
                                         weight * np.random.normal(1., 0.2),
                                         input_idx, input_spike_train)

        # Inhibitory input
        # input_idxs = cell.get_rand_idx_area_norm(section='allsec', nidx=20,
        #                                          z_min=cell.zmid[0]-100,
        #                                          z_max=cell.zmid[0]+100)
        # input_center = 15
        # for input_idx in input_idxs:
        #     input_spike_train = np.random.normal(input_center, 2.0, size=1)
        #     cell, synapse = make_synapse(cell,
        #                                  weight * np.random.normal(1., 0.2),
        #                                  input_idx, input_spike_train, e=-90)


    else:
        raise RuntimeError("Stimuli not recognized")
    return cell, synapse

def return_electrode_parameters():
    # Making x,y,z coordinates of three electrodes
    elec_z = np.linspace(-600, 0, 12)#np.linspace(-900, 0, 9)
    elec_x = np.ones(len(elec_z)) * (0)
    elec_y = np.zeros(len(elec_z))
    electrode_parameters = {
        'sigma': 0.3,       # extracellular conductivity
        'x': elec_x,        # x,y,z-coordinates of contact points
        'y': elec_y,
        'z': elec_z,
        'method': 'soma_as_point',
    }
    return electrode_parameters


def single_cell_compare(cell_number=1, celltype="almog",
                        conductance_type="passive",
                        input_type="pulse", plot=False):

    # print cell_number
    fig_name = 'swr_%s_%s_%s_%04d' % (input_type, celltype, conductance_type, cell_number)

    plt.seed((random_seed + random_seed_shift[celltype]) * cell_number)

    cell = return_cell(celltype, conductance_type, cell_number)

    # apic_idx = cell.get_closest_idx(x=0., z=-300, y=0.)
    cell, syn_input = make_input(cell, input_type, celltype)

    # print np.mean(cell.diam), np.median(cell.diam)
    cell.simulate(rec_imem=True, rec_vmem=True)

    electrode_parameters = return_electrode_parameters()

    # plt.close("all")
    # [plt.plot(cell.tvec, cell.vmem[idx, :]) for idx in range(cell.totnsegs)]
    # plt.show()

    # animate_vmem(cell, fig_name)

    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()

    LFP = 1000 * electrode.LFP

    if not os.path.isdir(join(root_folder, celltype, "EAPs")):
        os.mkdir(join(root_folder, celltype))
        os.mkdir(join(root_folder, celltype, "EAPs"))

    spike_time_idx = np.argmax(cell.somav)
    tlim = [cell.tvec[spike_time_idx] - 15, cell.tvec[spike_time_idx] + 15]
    spike_window_idx = [np.argmin(np.abs(cell.tvec - lim)) for lim in tlim]

    np.save(join(root_folder, celltype, "EAPs", "EAP_%s.npy" % fig_name),
            LFP[:, spike_window_idx[0]:spike_window_idx[1]])

    np.save(join(root_folder, celltype, "EAPs", "somav_%s.npy" % fig_name),
            cell.somav[spike_window_idx[0]:spike_window_idx[1]])

    np.save(join(root_folder, celltype, "EAPs", "imem_%s.npy" % fig_name),
            cell.imem[:, spike_window_idx[0]:spike_window_idx[1]])

    np.save(join(root_folder, celltype, "EAPs", "spiketime_%s.npy" % fig_name),
            cell.tvec[spike_time_idx])

    if plot or not cell_number % 10 or cell_number < 10:
        print "Vmem range: ", np.min(cell.vmem), np.max(cell.vmem)
        plot_single_cell_LFP(cell,
                             electrode_parameters, celltype,
                             fig_name)


def plot_single_cell_LFP(cell, electrode_parameters,
                         celltype, fig_name):


    ### MAKING THE ELECTRODE
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    electrode.calc_lfp()

    spike_time_idx = np.argmax(cell.somav)
    spike_time = cell.tvec[spike_time_idx]
    tlim = [cell.tvec[spike_time_idx] - 5, cell.tvec[spike_time_idx] + 10]

    ### PLOTTING THE RESULTS
    elec_idx_colors = {idx: plt.cm.Reds_r(1./(len(electrode.x) + 1) * idx) for idx in range(len(electrode.x))}

    plt.close('all')
    plt.figure(figsize=(16, 9))
    # Plotting the morphology
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.suptitle('Red: Extracellular recordings')
    ax_m = plt.subplot(142, aspect=1, xlabel='x [$\mu m$]', ylabel='z [$\mu m$]', ylim=[-900, 100], xlim=[-200, 200],
                xticks=[], frameon=False, yticks=[-800, -600, -400, -200, 0])
    [plt.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]], c='k', clip_on=False) for idx in xrange(cell.totnsegs)]
    plt.plot(cell.xmid[0], cell.zmid[0], 'o', c="k", ms=12)

    [plt.plot(electrode.x[idx], electrode.z[idx], 'D',  c=elec_idx_colors[idx], ms=8, clip_on=False)
     for idx in xrange(len(electrode.x))]

    ax_sp = plt.subplot(141,  xlabel='Time [ms]', ylabel='z [$\mu$m]', title="Single-cell input spike-trains",
                       xlim=tlim, sharey=ax_m, frameon=False)

    for syn_idx, syn in enumerate(cell.synapses):

        spikes = cell.sptimeslist[syn_idx]

        c = "r" if syn.kwargs["e"] > -60. else 'b'
        ax_sp.plot(spikes, np.ones(len(spikes)) * cell.zmid[syn.idx], '.', c=c)
        ax_m.plot(cell.xmid[syn.idx], cell.zmid[syn.idx], '*', ms=7, clip_on=False, c=c)
    # Plotting the extracellular potentials
    LFP = 1000 * (electrode.LFP - electrode.LFP[:, 0, None])

    dz = electrode.z[1] - electrode.z[0]


    ax_vsd2 = plt.subplot(143, title="somatic\nmembrane potential\nSpike time: {:0.2f} ms". format(spike_time), xlim=tlim, ylim=[-80, 50])
    ax_vsd2.plot(cell.tvec, cell.somav)
    ax_vsd2.axvline(spike_time, ls='--', c='gray')
    # img = ax_vsd2.imshow(vsd, extent=[0, cell.tvec[-1], np.min(z_box) - dz/2, np.max(z_box) + dz/2],
    #                     interpolation='nearest', origin='lower', cmap=plt.cm.Reds,
    #                      vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar(img, shrink=0.5, label='mV')
    # ax_vsd2.plot([cell.tvec[-1] - 10, cell.tvec[-1]],
    #             [np.min(electrode.z) - 20, np.min(electrode.z) - 20], lw=4, c='k', clip_on=False)
    # ax_vsd2.text(cell.tvec[-1] - 10, np.min(electrode.z) - 50, '10 ms')
    # ax_vsd2.axis("auto")

    ax_lfp = plt.subplot(144, frameon=False, yticks=[-800, -600, -400, -200, 0], sharey=ax_m,
                         title="Extracellular potential", xlim=tlim)
    # vmax = np.max(np.abs(LFP)) / 10
    # vmin = -vmax
    # img = ax_lfp.imshow(LFP, extent=[tlim[0], tlim[1], np.min(electrode.z) - dz/2, np.max(electrode.z) + dz/2],
    #                     interpolation='nearest', origin='lower', cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax)
    #
    # cbar = plt.colorbar(img, shrink=0.5, label='$\mu$V')

    normalize = np.max(np.abs((LFP[:, :] - LFP[:, 0, None])))
    for idx in range(len(electrode.z)):
        y = electrode.z[idx] + (LFP[idx] - LFP[idx, 0]) / normalize * dz
        ax_lfp.plot(cell.tvec, y, lw=1, c='k', clip_on=True)

    ax_lfp.axvline(spike_time, ls='--', c='gray')

    ax_lfp.plot([tlim[1], tlim[1]],
                [np.min(electrode.z) +dz, np.min(electrode.z)], lw=4, c='k', clip_on=False)
    ax_lfp.text(tlim[1] + 1, np.min(electrode.z) + dz/2, '{:.2f} $\mu$V'.format(normalize), clip_on=False)
    # ax_lfp.axis("auto")

    plt.savefig(join(root_folder, celltype, '%s.png' % fig_name))
    # plt.show()

if __name__ == '__main__':

    conductance_type = "active"
    if len(sys.argv) == 1:
        celltype = ['hbp_L5_TTPC2_cADpyr232_1', "hbp_L4_SS_cADpyr230_1", "hbp_L4_SBC_bNAC219_1"][1]
        input_type = ["waves"][0]
        initialize_population(10000, celltype)
        single_cell_compare(celltype=celltype, input_type=input_type,
                            cell_number=3, plot=True,
                            conductance_type=conductance_type)
        # single_cell_ca_spike(conductance_type='passive')
    else:
        single_cell_compare(
                            celltype=sys.argv[1],
                            input_type=sys.argv[2],
                            cell_number=int(sys.argv[3]),
                            conductance_type=conductance_type)
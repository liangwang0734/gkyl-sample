import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
show_on_screen = True
if not show_on_screen:
    mpl.use("agg")  # if not showing on screen, agg is faster

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import h5py
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
from matplotlib.ticker import MaxNLocator
import os.path
from time import time as now

"""
Attempts to make things faster or to handle huge data:
- Convert observation data to binary format for faster repeated access, if no
    converted file exists.
- Extract model data from local region for faster repeated access, if no
    extracted file exists.
- Only sample points between tstart and tend.
- Option read_all_to_mem to read entire data block into memory or do i/o
    directly on disk. Works good for small files. Does not work well with large
    dataset as slicing over component and z index can be very slow.
- If read_all_to_mem is False, then we can use option extract_step to specify
    each time how many x indices to step over. We iterate over jx indices, and
    first read data block of index [jx:jx+extract_step, iy, :, :] into memory (
    very slow), and then extract the component and iz range data into files
    (fast). This works well with large files. Use as large extrct_step as
    possible.
"""

# original data files
folder = "3D-10mom_3_Xianzhe_InnerBcs_highres_1_3/"
folder = "./"
folder_obs = "./"
fname_obs = folder_obs + 'MAGMSOSCI08280_V08.TAB'
fname0 = folder + "mercury_v2_staticField.h5"
fname1 = folder + "mercury_v2_q_{}.h5"
frames = [6]
# observation data file saved in binary format for faster access
fname_obs_bin = folder_obs + "MAGMSOSCI08280_V08.npz"
# extracted simulation data files; each B component in a separate file
fname_B1_fmt = folder + "mercury_v2_compare_B1_{}.h5"
fname_B0_fmt = folder + "mercury_v2_compare_B0.h5"

tick_label_fmt = mdates.DateFormatter("%H:%M:%S")
tstart = pd.to_datetime(pd.datetime(2008, 10, 6, 7, 0, 0))
tend = pd.to_datetime(pd.datetime(2008, 10, 6, 10, 0, 0))
# slicing stride for sampling points along the trajectory
# larger stride means fewer sampling points
stride = 1000
# mode can be "nearest" or "map_coordinates"
mode = "map_coordinates"

# output figure
figname = 'output_{}.png'.format(mode)

# constants
R_km = 2439.7
iR_km = 1. / R_km
R = R_km * 1e3
iR = 1. / R

# observation data i/o options
# convert observation date from ascii (TAB) to binary, if not exists
try_convert_obs_bin = True
# read observation data saved in the binary file, if exists
try_read_obs_bin = True

# gkeyll data extraction options
# extract background B field in local region
force_extract_B0 = False
# extract B1 in local region
force_extract_B1 = False
Bx_idx0 = 3
Bx_idx = 23
# default extract options
extract_kwargs = dict(
    xmin=-10 * R,
    xmax=10 * R,
    xstep=1,
    ymin=-8 * R,
    ymax=10 * R,
    ystep=1,
    zmin=-0.5 * R,
    zmax=0.5 * R,
    zstep=1,
    dtype="float32",  # use float16 if mem overflows
    coeff=1e9,  # nano-tesla
    read_all_to_mem=False,  # set False if mem overflows
    extract_step=10,  # larger, faster, but might overflow
)

# gkeyll data sampling options
# read entire component data into memory before sampling
read_all_to_mem = True
if mode == "map_coordinates":
    # set order = 0 to use nearest point
    map_kwargs = dict(order=3)

# READ OBSERVED B FIELD
print("***** Reading observation data...")
exists = os.path.exists(fname_obs_bin)
if try_read_obs_bin and exists:
    arr = np.load(fname_obs_bin)
    x = arr["x"]
    y = arr["y"]
    z = arr["z"]
    Bx = arr["Bx"]
    By = arr["By"]
    Bz = arr["Bz"]
    B = arr["B"]
    time = pd.to_datetime(arr["time"])
else:
    arr = np.loadtxt(fname_obs)
    x = arr[:, 6] * iR_km
    y = arr[:, 7] * iR_km
    z = arr[:, 8] * iR_km
    Bx = arr[:, 9]
    By = arr[:, 10]
    Bz = arr[:, 11]
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    year = arr[:, 0].astype(np.int64)
    day_of_year = arr[:, 1].astype(np.int64)
    hour = arr[:, 2].astype(np.int64)
    minute = arr[:, 3].astype(np.int64)
    second = arr[:, 4]
    time0_str = '{} {} {} {} {}'.format(year[0], day_of_year[0], hour[0],
                                        minute[0], second[0])
    time0 = pd.datetime.strptime(time0_str, '%Y %j %H %M %S.%f')
    dt_in_s = (
        (day_of_year - day_of_year[0]) * 24.0 * 3600 + (hour - hour[0]) * 3600
        + (minute - minute[0]) * 60 + second - second[0])
    dt_in_s += (time0 - pd.datetime(1970, 1, 1, 0, 0, 0)).total_seconds()
    time = pd.to_datetime(dt_in_s, unit='s')

if try_convert_obs_bin and not exists:
    print("convert observation data to {}".format(fname_obs_bin))
    np.savez(fname_obs_bin, x=x, y=y, z=z, Bx=Bx, By=By, Bz=Bz, B=B, time=time)
print("time[0]:  {}".format(time[0]))
print("time[-1]: {}".format(time[-1]))

mask = (time < tend) & (time > tstart)
x = x[mask]
y = y[mask]
z = z[mask]
Bx = Bx[mask]
By = By[mask]
Bz = Bz[mask]
B = B[mask]
time = time[mask]
print("flyby x min {:10g}, max {:10g}".format(x.min(), x.max()))
print("flyby y min {:10g}, max {:10g}".format(y.min(), y.max()))
print("flyby z min {:10g}, max {:10g}".format(z.min(), z.max()))

# EXTRACT REGIONAL DATA
print("\n***** Extracting model data...")


def adjust_idx(ix_cc, nx):
    """
    Make sure slice indices are within bound.

    Returns:
        (ix_cc, ix_nc)
    """
    start, stop, step = ix_cc.indices(nx)
    ix_cc = slice(start, stop, step)
    start, stop, step = slice(start, stop + step, step).indices(nx)
    ix_nc = slice(start, stop, step)
    return ix_cc, ix_nc


def extract3d(filename_in,
              filename_out,
              comp,
              xmin,
              xmax,
              xstep,
              ymin,
              ymax,
              ystep,
              zmin,
              zmax,
              zstep,
              read_all_to_mem=True,
              extract_step=100,
              coeff=1,
              **kwargs):
    """
    Extract one component in a local region.

    Arguments:
    read_all_to_mem: True or False. If True, original dataset is loaded into
        memory for fast extraction; If False, extract from hdf5 dataset on disk
        directly so that very large files can be handled.
    extract_step: If read_all_to_mem is False, then we iterate x indices to read
        all components in many yz planes. This argument sets how many x indices
        we step over each time, i.e., each time we read data in x index range jx
        to jx+extract_out.
    coeff: Multiplication coefficient.
    kwargs: Arguments passed to h5py.create_dataset. Examples are:
        dtype: For example, "float32". Note that a lower precision dtype use
            less space but might truncate tiny/huge values. Then it might be
            useful to adjust multiplication coeff so that most values are well
            represented.
        chunks: ...
    """
    file_in = h5py.File(filename_in, 'r')
    file_out = h5py.File(filename_out, 'w')
    # grid
    sg_in = file_in["StructGrid"]
    vsKind = sg_in.attrs["vsKind"].decode("UTF-8")
    if vsKind in ["structured"]:
        xnc = sg_in[:, 0, 0, 0]
        ync = sg_in[0, :, 0, 1]
        znc = sg_in[0, 0, :, 2]
    elif vsKind in ["uniform"]:
        lo = np.array(sg_in.attrs["vsLowerBounds"])
        up = np.array(sg_in.attrs["vsUpperBounds"])
        ncells = np.array(sg_in.attrs["vsNumCells"])
        xnc, ync, znc = [
            np.linspace(lo[d], up[d], ncells[d]) for d in range(3)
        ]
    x = 0.5 * (xnc[1:] + xnc[:-1])
    y = 0.5 * (ync[1:] + ync[:-1])
    z = 0.5 * (znc[1:] + znc[:-1])
    # slices
    ix = slice(np.searchsorted(x, xmin), np.searchsorted(x, xmax), xstep)
    iy = slice(np.searchsorted(y, ymin), np.searchsorted(y, ymax), ystep)
    iz = slice(np.searchsorted(z, zmin), np.searchsorted(z, zmax), zstep)
    nx = x.shape[0]
    ny = y.shape[0]
    nz = z.shape[0]
    ix, ixg = adjust_idx(ix, nx)
    iy, iyg = adjust_idx(iy, ny)
    iz, izg = adjust_idx(iz, nz)
    print("extraction slices: ix {} iy {} iz {}".format(ix, iy, iz))
    nx = (ix.stop - ix.start) // ix.step
    ny = (iy.stop - iy.start) // iy.step
    nz = (iz.stop - iz.start) // iz.step
    print("extraction size: nx {} ny {} nz {}".format(nx, ny, nz))
    # field
    sgf_in = file_in["StructGridField"]
    if read_all_to_mem:
        kwargs.setdefault("dtype", sgf_in.dtype)
        data = sgf_in[ix, iy, iz, :] * coeff
        Bx_out = file_out.create_dataset("Bx", data=data[..., comp], **kwargs)
        By_out = file_out.create_dataset(
            "By", data=data[..., comp + 1], **kwargs)
        Bz_out = file_out.create_dataset(
            "Bz", data=data[..., comp + 2], **kwargs)
    else:
        nx = (ix.stop - ix.start) // ix.step
        ny = (iy.stop - iy.start) // iy.step
        nz = (iz.stop - iz.start) // iz.step
        kwargs.setdefault("dtype", sgf_in.dtype)
        Bx_out = file_out.create_dataset("Bx", shape=(nx, ny, nz), **kwargs)
        By_out = file_out.create_dataset("By", shape=(nx, ny, nz), **kwargs)
        Bz_out = file_out.create_dataset("Bz", shape=(nx, ny, nz), **kwargs)
        jxs = range(ix.start, ix.stop, ix.step)
        njx = len(jxs)
        t00 = now()
        for jx_out, jx in enumerate(jxs):
            if jx_out % extract_step == 0 and jx_out + extract_step <= njx:
                t0 = now()
                print("jx {}:{}, jx_out {}:{} out of {}".format(
                    jx, jx + extract_step, jx_out, jx_out + extract_step, nx))
                tmp = sgf_in[jx:jx + extract_step, iy, :, :] * coeff
                print("    extract file to mem took {}s".format(now() - t0))
                t0 = now()
                Bx_out[jx_out:jx_out + extract_step, ...] = tmp[..., iz, comp]
                By_out[jx_out:jx_out + extract_step, ...] = tmp[..., iz, comp +
                                                                1]
                Bz_out[jx_out:jx_out + extract_step, ...] = tmp[..., iz, comp +
                                                                2]
                print("    extract mem to file took {}s".format(now() - t0))
                del(tmp)  # FIXME does this really free the memory?
            elif (jx_out +
                  extract_step) > njx and jx_out % extract_step == 0:
                print("jx {}:{}, jx_out {}: out of {}".format(
                    jx, jxs[-1] + 1, jx_out, nx))
                tmp = sgf_in[jx:jxs[-1] + 1, iy, :, :] * coeff
                Bx_out[jx_out:, ...] = tmp[..., iz, comp]
                By_out[jx_out:, ...] = tmp[..., iz, comp + 1]
                Bz_out[jx_out:, ...] = tmp[..., iz, comp + 2]
        print("extraction took {}s".format(now() - t00))
    for attr in ['vsType', 'vsMesh', 'vsCentering']:
        for dset in [Bx_out, By_out, Bz_out]:
            dset.attrs[attr] = sgf_in.attrs[attr]
    # mesh
    sg_out = file_out.create_group("StructGrid")
    sg_out.attrs["vsType"] = np.string_("mesh")
    sg_out.attrs["vsKind"] = np.string_("rectilinear")
    sg_out.create_dataset("axis0", data=xnc[ixg])
    sg_out.create_dataset("axis1", data=ync[iyg])
    sg_out.create_dataset("axis2", data=znc[izg])
    # time
    file_out.create_group('timeData')
    for attr in ['vsType', 'vsTime', 'vsStep']:
        file_out['timeData'].attrs[attr] = file_in['timeData'].attrs[attr]
    file_out.close()
    file_in.close()


fname_out = fname_B0_fmt
exists = os.path.exists(fname_out)
if force_extract_B0 or not exists:
    print("\nextracting to {}".format(fname_out))
    extract3d(fname0, fname_out, Bx_idx0, **extract_kwargs)
else:
    print("{} exists; skipping".format(fname_out))

for frame in frames:
    fname_out = fname_B1_fmt.format(frame)
    exists = os.path.exists(fname_out)
    if force_extract_B1 or not exists:
        print("\nextracting to {}".format(fname_out))
        extract3d(
            fname1.format(frame), fname_out, comp=Bx_idx, **extract_kwargs)
    else:
        print("{} exists; skipping".format(fname_out))

print("\n***** Sampling model data...")
# READ SIMULATION GRID
with h5py.File(fname_B0_fmt) as fp:
    sg = fp["StructGrid"]
    xnc = sg["axis0"][:] * iR
    ync = sg["axis1"][:] * iR
    znc = sg["axis2"][:] * iR
    xcc = 0.5 * (xnc[1:] + xnc[:-1])
    ycc = 0.5 * (ync[1:] + ync[:-1])
    zcc = 0.5 * (znc[1:] + znc[:-1])

print("extracted region x min {:10g}, max {:10g}".format(xcc.min(), xcc.max()))
print("extracted region y min {:10g}, max {:10g}".format(ycc.min(), ycc.max()))
print("extracted region z min {:10g}, max {:10g}".format(zcc.min(), zcc.max()))
coords = [xcc, ycc, zcc]
coords_t = np.array([x[::stride], y[::stride], z[::stride]])


# READ B FIELD FROM SIMULATION AND MAKE PLOTS
def prep_nearest(coords, coords_t, **kwargs):
    npts = coords_t.shape[1]
    idx = np.empty([3, npts], dtype=int)
    for d in range(3):
        x = coords[d]
        xt = coords_t[d]
        nx, = x.shape
        idx[d, :] = np.clip(np.searchsorted(x, xt, **kwargs), 0, nx - 1)
    return idx


def interp_nearest(data, idx, read_all_to_mem=True, **kwargs):
    npts = idx.shape[1]
    pts = np.empty([npts])
    if read_all_to_mem:
        data = data[...]
    for ipt in range(npts):
        pts[ipt] = data[tuple(idx[:, ipt])]
    return np.asarray(pts)


def prep_map_coordinates(coords, coords_t, **kwargs):
    xti = []
    kind = kwargs.pop("kind", "linear")
    for d in range(3):
        x = coords[d]
        # xi: image coordinates of field, i.e., 0 to n-1; ints
        xi = np.arange(len(x))
        xt = np.clip(coords_t[d], x.min(), x.max())
        # xti: image coordiantes of sampling points; floats
        xti.append(interpolate.interp1d(x, xi, kind=kind)(xt))
    return np.vstack(xti)


def interp_map_coordinates(data, xti, read_all_to_mem=True, **kwargs):
    if read_all_to_mem:
        data = data[...]
    return ndimage.map_coordinates(data, xti, **kwargs)


if mode == "map_coordinates":
    xti = prep_map_coordinates(coords, coords_t)
    interp = interp_map_coordinates
    kwargs = dict(xti=xti, read_all_to_mem=read_all_to_mem)
    kwargs.update(map_kwargs)
elif mode == "nearest":
    idx = prep_nearest(coords, coords_t)
    interp = interp_nearest
    kwargs = dict(idx=idx, read_all_to_mem=read_all_to_mem)

with h5py.File(fname_B0_fmt) as fp:
    Bx0 = interp(fp['Bx'], **kwargs)
    By0 = interp(fp['By'], **kwargs)
    Bz0 = interp(fp['Bz'], **kwargs)
Bmag0 = np.sqrt(Bx0**2 + By0**2 + Bz0**2)

# TODO update figure data instead of redraw everything
for frame in frames:
    with h5py.File(fname_B1_fmt.format(frame)) as fp:
        Bx1 = interp(fp['Bx'], **kwargs)
        By1 = interp(fp['By'], **kwargs)
        Bz1 = interp(fp['Bz'], **kwargs)

    Bxt = Bx0 + Bx1
    Byt = By0 + By1
    Bzt = Bz0 + Bz1
    Bmagt = np.sqrt(Bxt**2 + Byt**2 + Bzt**2)

    print("\n***** Making plot...")
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
    for i, ax in enumerate(axs):
        ax.plot(
            time, [Bx, By, Bz, B][i],
            c='k',
            lw=2,
            ls='-',
            alpha=0.75,
            label=['$B_x$', '$B_y$', '$B_z$', '$|B|$'][i] + ' - MAG')
        if i == 3:
            ax.plot(
                time[::stride], [Bx0, By0, Bz0, Bmag0][i],
                c='b',
                lw=2,
                ls='--',
                alpha=0.75,
                label=['$B_{x0}$', '$B_{y0}$', '$B_{z0}$', '$|B_0|$'][i])
        ax.plot(
            time[::stride], [Bxt, Byt, Bzt, Bmagt][i],
            c='r',
            lw=2,
            ls='-',
            alpha=0.75,
            label=['$B_x$', '$B_y$', '$B_z$', '$|B|$'][i] + ' - Gkeyll')
        ax.xaxis.set_major_formatter(tick_label_fmt)
        ax.legend()
        # ax.axhline(0, color='k', linestyle='dashed')  # mark horizontal axis
        ax.grid()
        ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
        ax.set_xlim(tstart, tend)
    axs[-1].set_xlabel('Time (UT) in October 6 , 2008')
    fig.subplots_adjust(hspace=0)

    plt.savefig(figname, bbox_inches='tight', dpi=320)
    if show_on_screen:
        plt.show()
    else:
        plt.close()

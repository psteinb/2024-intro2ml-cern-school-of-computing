import matplotlib.pyplot as plt
import numpy as np

np_cat = np.concatenate
np_permute = np.transpose


def np_relu(arr):
    zeros = np.zeros_like(arr)
    return np.maximum(arr, zeros)


def np_max_pool1d(arr, kernel_size=2, stride=1):
    oshape = arr.shape
    nshape = (*oshape[:-1], oshape[-1] // kernel_size)
    value = arr.reshape((-1, kernel_size)).max(axis=-1)
    return value.reshape(nshape)


def positions_to_sequences(
    tr=None, bx=None, noise_level=0.3, seq_length=100, rng=np.random.default_rng()
):
    """
    tr: positions of triangles
    bx: positions of boxes
    noise_level: uniform noise to add
    seq_length: length of sequence
    """

    st = np.arange(seq_length).astype(np.float32)
    st = st[None, :, None]
    tr = tr[:, None, :, :]
    bx = bx[:, None, :, :]

    xtr = np_relu(
        tr[..., 1]
        - np_relu(np.abs(st - tr[..., 0]) - 0.5) * 2 * tr[..., 1] / tr[..., 2]
    )
    xbx = (
        np.sign(
            np_relu(
                bx[..., 1] - np.abs((st - bx[..., 0]) * 2 * bx[..., 1] / bx[..., 2])
            )
        )
        * bx[..., 1]
    )

    x = np_cat((xtr, xbx), 2)

    u = np_max_pool1d(np.transpose(np.sign(x), axes=(0, 2, 1)), kernel_size=2, stride=1)
    u = np.transpose(u, axes=(0, 2, 1))

    collisions = (u.sum(2) > 1).max(1)
    y = x.max(2)

    return y + rng.uniform(size=y.shape) * noise_level - noise_level / 2, collisions


def generate_sequences(
    nb,
    seq_length=100,
    seq_height_min=1.0,
    seq_height_max=25.0,
    seq_width_min=5.0,
    seq_width_max=11.0,
    group_by_locations=False,
    rng=np.random.default_rng(),
    out_dtype= np.float32
):
    # Position / height / width

    tr = np.empty(shape=(nb, 2, 3))

    tr[:, :, 0] = rng.uniform(
        low=seq_width_max / 2, high=seq_length - seq_width_max / 2, size=tr.shape[:-1]
    )
    tr[:, :, 1] = rng.uniform(
        low=seq_height_min, high=seq_height_max, size=tr.shape[:-1]
    )
    tr[:, :, 2] = rng.uniform(low=seq_width_min, high=seq_width_max, size=tr.shape[:-1])

    bx = np.empty(shape=(nb, 2, 3))

    bx[:, :, 0] = rng.uniform(
        low=seq_width_max / 2, high=seq_length - seq_width_max / 2, size=tr.shape[:-1]
    )
    bx[:, :, 1] = rng.uniform(
        low=seq_height_min, high=seq_height_max, size=tr.shape[:-1]
    )
    bx[:, :, 2] = rng.uniform(low=seq_width_min, high=seq_width_max, size=tr.shape[:-1])

    if group_by_locations:
        a = np_cat((tr, bx), 1)
        v = a[:, :, 0].sort(1)[:, 2:3]
        mask_left = (a[:, :, 0] < v).astype(np.float32)
        h_left = (a[:, :, 1] * mask_left).sum(1) / 2
        h_right = (a[:, :, 1] * (1 - mask_left)).sum(1) / 2
        valid = (h_left - h_right).abs() > 4
    else:
        valid = (np.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4) & (
            np.abs(tr[:, 0, 1] - tr[:, 1, 1]) > 4
        )

    input, collisions = positions_to_sequences(tr, bx, seq_length=seq_length, rng=rng)

    if group_by_locations:
        a = np_cat((tr, bx), 1)
        v = a[:, :, 0].sort(1)[:, 2:3]
        mask_left = (a[:, :, 0] < v).astype(np.float32)
        h_left = (a[:, :, 1] * mask_left).sum(1, keepdims=True) / 2
        h_right = (a[:, :, 1] * (1 - mask_left)).sum(1, keepdims=True) / 2
        a[:, :, 1] = mask_left * h_left + (1 - mask_left) * h_right
        tr, bx = a.split(2, 1)
    else:
        tr[:, :, 1:2] = tr[:, :, 1:2].mean(1, keepdims=True)
        bx[:, :, 1:2] = bx[:, :, 1:2].mean(1, keepdims=True)

    targets, _ = positions_to_sequences(tr, bx, seq_length=seq_length, rng=rng)

    valid = valid & ~collisions
    tr = tr[valid]
    bx = bx[valid]
    input = input[valid][:, None, :]
    targets = targets[valid][:, None, :]

    if input.shape[0] < nb:
        input2, targets2, tr2, bx2 = generate_sequences(
            nb - input.shape[0],
            seq_length,
            seq_height_min,
            seq_height_max,
            seq_width_min,
            seq_width_max,
            group_by_locations,
            rng,
        )
        input = np_cat((input, input2), 0)
        targets = np_cat((targets, targets2), 0)
        tr = np_cat((tr, tr2), 0)
        bx = np_cat((bx, bx2), 0)

    return input.astype(out_dtype), targets.astype(out_dtype), tr.astype(out_dtype), bx.astype(out_dtype)


def save_sequence_images(
    filename, sequences, tr=None, bx=None, title="", seq_length=100, seq_height_max=500
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim(0, seq_length)
    ax.set_ylim(-1, seq_height_max + 4)

    for u in sequences:
        ax.plot(np.arange(u[0].shape[0]) + 0.5, u[0], color=u[1], label=u[2])
        ax.set_title(title)

    ax.legend(frameon=False, loc="upper left")

    delta = -1.0
    if tr is not None:
        ax.scatter(
            tr[:, 0],
            np.full((tr.shape[0],), delta),
            color="black",
            marker="^",
            clip_on=False,
        )

    if bx is not None:
        ax.scatter(
            bx[:, 0],
            np.full((bx.shape[0],), delta),
            color="black",
            marker="s",
            clip_on=False,
        )

    fig.savefig(filename, bbox_inches="tight")

    plt.close("all")

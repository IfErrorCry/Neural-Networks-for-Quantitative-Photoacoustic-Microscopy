import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve
from scipy.sparse import block_diag

# --- Henyey-Greenstein phase function ---
def henyey_greenstein(cos_theta, g):
    return (1 - g**2) / (1 + g**2 - 2 * g * cos_theta)**(3/2)

# --- Function to compute directional gradients ---
def compute_directional_gradients_2d(nx, ny, dx, dy):
    N = nx * ny  # Total number of grid nodes

    # --- Main directions ---
    # Right (M_x)
    main_diag_x = np.ones(N)
    lower_diag_x = -np.ones(N - 1)
    lower_diag_x[np.arange(1, N) % nx == 0] = 0
    M_x = diags([main_diag_x, lower_diag_x], [0, -1], format='csr') / dx

    # Left (M_-x)
    main_diag_neg_x = np.ones(N)
    upper_diag_x = -np.ones(N - 1)
    upper_diag_x[np.arange(N - 1) % nx == (nx - 1)] = 0
    M_neg_x = diags([main_diag_neg_x, upper_diag_x], [0, 1], format='csr') / dx

    # Up (M_y)
    main_diag_y = np.ones(N)
    lower_diag_y = -np.ones(N - ny)
    # lower_diag_y[:nx] = 0
    M_y = diags([main_diag_y, lower_diag_y], [0, -nx], format='csr') / dy

    # Down (M_-y)
    main_diag_neg_y = np.ones(N)
    upper_diag_y = -np.ones(N - nx)
    # upper_diag_y[-nx:] = 0
    M_neg_y = diags([main_diag_neg_y, upper_diag_y], [0, nx], format='csr') / dy

    # --- Diagonal directions ---
    M_diag_ur = (M_x + M_y) / np.sqrt(2)         # Up-Right (1,1)/√2
    M_diag_ul = (M_neg_x + M_y) / np.sqrt(2)     # Up-Left (-1,1)/√2
    M_diag_dl = (M_neg_x + M_neg_y) / np.sqrt(2) # Down-Left (-1,-1)/√2
    M_diag_dr = (M_x + M_neg_y) / np.sqrt(2)     # Down-Right (1,-1)/√2 


    return M_x, M_diag_ur, M_y, M_diag_ul, M_neg_x, M_diag_dl, M_neg_y, M_diag_dr


# --- Simulation parameters ---
nx, ny = 50, 50
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# absorption_coeff = 1.0  # Absorption coefficient
scattering_coeff = 50  # Scattering coefficient
g = 0.85 # Scattering anisotropy


# --- Directions encoded using basis (+x, -x, +y, -y) ---
directions = np.array([
    [1, 0, 0, 0],  # (1, 0)
    [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],  # (1/√2, 0, 1/√2, 0)
    [0, 0, 1, 0],  # (0, 1)
    [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],  # (0, 1/√2, 1/√2, 0)
    [0, 1, 0, 0],  # (0, -1)
    [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)],  # (0, 1/√2, 0, 1/√2)
    [0, 0, 0, 1],  # (-1, 0)
    [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],  # (1/√2, 0, 0, 1/√2)
])


direction_new =  np.array([
    [1, 0],  # (1, 0)
    [1 / np.sqrt(2), 1 / np.sqrt(2)],  # (1/√2, 0, 1/√2, 0)
    [0, 1],  # (0, 1)
    [-1 / np.sqrt(2), 1 / np.sqrt(2)],  # (0, 1/√2, 1/√2, 0)
    [-1, 0],  # (0, -1)
    [-1 / np.sqrt(2),-1 / np.sqrt(2)],  # (0, 1/√2, 0, 1/√2)
    [0, -1],  # (-1, 0)
    [1 / np.sqrt(2), - 1 / np.sqrt(2)],  # (1/√2, 0, 0, 1/√2)
])
n_dirs = direction_new.shape[0]

k_matrix = np.zeros((n_dirs, n_dirs))
for i in range(n_dirs):
    for j in range(n_dirs):
        cos_theta = np.dot(direction_new[i], direction_new[j])
        k_matrix[i, j] = henyey_greenstein(cos_theta, g)
k_matrix /= np.sum(k_matrix, axis=1, keepdims=True)



# --- Create spatial grid ---
#x = np.linspace(0, Lx, nx)
#y = np.linspace(0, Ly, ny)
x = np.linspace(-Lx/2, Lx/2, nx)
y = np.linspace(-Ly/2, Ly/2, ny)
X, Y = np.meshgrid(x, y)

# --- Light source (Gaussian distribution) ---
q = 50 * np.exp(-100 * ((1/500)* X**2 + (Y - 0.45)**2))


# --- Get directional gradient operators ---
A_x, A_diag_ur, A_y, A_diag_ul, A_neg_x, A_diag_dl, A_neg_y, A_diag_dr = compute_directional_gradients_2d(nx, ny, dx, dy)

# --- Build system matrix ---
N = nx * ny

# --- Transport operators for each direction ---
A_directions = []
for vx, v_negx, vy, v_negy in directions:
    A_dir = (
        kron(diags([vx], [0]), A_x) +
        kron(diags([v_negx], [0]), A_neg_x) +
        kron(diags([vy], [0]), A_y) +
        kron(diags([v_negy], [0]), A_neg_y)
    )
    A_directions.append(A_dir)

# --- Spatially varying absorption coefficient ---
# abs_coeff_map = np.full_like(X, 0.5)

# abs_coeff_map = (0.5 + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y))* 10
w_sigma = 0.3
abs_coeff_map = 0.2 * np.exp(-((X - 0.2)**2 + Y**2)/ (2 * w_sigma**2))

# abs_coeff_map = np.zeros_like(X)
# abs_coeff_map[(X**2 + Y**2) < 0.1**2] = 10.0

"""
abs_coeff_map = 0.1 * np.zeros_like(X)

# Источник 1 (внизу слева)
abs_coeff_map[((X - 0.3)**2 + (Y - 0.3)**2) < 0.04**2] = 8.0

# Источник 2 (вверху справа)
abs_coeff_map[((X + 0.3)**2 + (Y + 0.3)**2) < 0.04**2] = 5.0

"""


#abs_coeff_map = np.ones_like(X)
#abs_coeff_map[(X-0.5)**2 + (Y-0.5)**2 < 0.1**2] = 10.0
absorption_coeff_vector = abs_coeff_map.flatten()

# Combine all directional transport matrices
# A_directional = kron(identity(n_dirs), sum(A_directions))
A_directional = block_diag(A_directions, format='csr')

# --- Absorption and scattering operator ---
# A_abs_scatter = kron(identity(n_dirs), diags([absorption_coeff + scattering_coeff], [0], shape=(N, N)))
A_abs_scatter = kron(identity(n_dirs), diags([absorption_coeff_vector + scattering_coeff], [0], shape=(N, N)))

# --- Scattering redistribution operator ---
A_scatter = kron(k_matrix, diags([scattering_coeff], [0], shape=(N, N)))

# --- Final system matrix ---
A_total = A_directional + A_abs_scatter - A_scatter

#print("scater", k_matrix)
#print(diags([scattering_coeff], [0], shape=(N, N)))



def make_directional_source(q_base, direction_index, n_dirs):
    """
    Creates a source vector where the light is injected only in one direction.
    """
    q_vector = np.zeros((n_dirs, *q_base.shape))
    q_vector[direction_index] = q_base
    return q_vector.reshape(-1)


def plot_heatmap(matrix, title):
    """
    Visualize a sparse matrix as a heatmap.
    Converts it to dense and crops to size×size.
    """
    dense = matrix.toarray()[:100, :100]
    
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(dense, cmap='viridis', interpolation='nearest', origin='upper')
    plt.title(f"Heatmap: {title}", fontsize=16, weight='bold', pad=15)
    plt.colorbar(heatmap, label='Intensity', orientation='vertical')
    plt.grid(False)
    plt.tick_params(axis='both', which='both', length=0)
    plt.tight_layout()
    plt.show()


# plot_heatmap(A_directional, "A_directional")
# plot_heatmap(A_abs_scatter, "A_abs_scatter")
# plot_heatmap(A_scatter, "A_scatter")
# plot_heatmap(A_total, "A_total")

def print_sparse_matrix(matrix, name):
    matrix = matrix.tocsr()  # in index format
    dense = matrix.toarray()  # in numpy form
    print(f"\n{name} ({dense.shape[0]}x{dense.shape[1]}):")
    print(dense)


# print_sparse_matrix(A_directional, "A_directional")
# print_sparse_matrix(A_abs_scatter, "A_abs_scatter")
# print_sparse_matrix(A_scatter, "A_scatter")
# print_sparse_matrix(A_total, "A_total")

# --- Solve the system ---
# q_vector = np.concatenate([q.flatten(), np.zeros((n_dirs - 1) * N)])
# Phi_flat = spsolve(A_total, q_vector)  # Solution as flat vector
# Phi = Phi_flat.reshape((n_dirs, ny, nx))  # Reshape to 3D
# Phi_total = Phi.sum(axis=0)  # Sum over all directions (fluence)

"""
# --- Visualization ---
plt.imshow(Phi_total, extent=(0, Lx, 0, Ly), origin='lower', cmap='hot')
plt.colorbar(label='Total Intensity Φ')
plt.title('Intensity Distribution (DOM, Henyey-Greenstein)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# --- Plot directional intensities (Φ_i for each direction) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
direction_labels = [
    "→", "↗", "↑", "↖", 
    "←", "↙", "↓", "↘"
]

for i in range(n_dirs):
    ax = axes[i // 4, i % 4]
    im = ax.imshow(Phi[i], extent=(0, Lx, 0, Ly), origin='lower', cmap='plasma')
    ax.set_title(f"Direction {i+1}: {direction_labels[i]}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Directional Intensities Φᵢ (per direction)", fontsize=16, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
"""

""""
Phi_total_all = np.zeros((ny, nx))

fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
fig.suptitle('Φ_total from Directional Sources (DOM)', fontsize=20)

for i in range(n_dirs):
    q_vector = make_directional_source(q, direction_index=i, n_dirs=n_dirs)
    Phi_flat = spsolve(A_total, q_vector)
    Phi = Phi_flat.reshape((n_dirs, ny, nx))
    Phi_total = Phi.sum(axis=0)
    Phi_total_all += Phi_total

    ax = axes.flat[i]
    im = ax.imshow(Phi_total, extent=(0, Lx, 0, Ly), origin='lower', cmap='hot')
    ax.set_title(f"Direction {i + 1}", fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.85, pad=0.02)
cbar.set_label('Φ_total')

plt.show()"""

""""
plt.figure(figsize=(8, 6))
plt.imshow(Phi_total_all, extent=(0, Lx, 0, Ly), origin='lower', cmap='hot')
plt.title('Φ_total from All Directions (Sum)', fontsize=16)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Φ_total (summed)')
plt.tight_layout()
plt.show()
"""

q_vector = make_directional_source(q, direction_index=6, n_dirs=n_dirs)
"""
q_vector = np.zeros((n_dirs, *q.shape))
for i in range(n_dirs):
    q_vector[i] = q
q_vector = q_vector.reshape(-1)
"""
Phi_flat = spsolve(A_total, q_vector)
Phi = Phi_flat.reshape((n_dirs, ny, nx))
Phi_total = Phi.sum(axis=0)

plt.figure(figsize=(8, 6))
plt.imshow(Phi_total, extent=(-Lx/2, Lx/2, -Ly/2, Ly/2), origin='lower', cmap='hot')
plt.title('Intensity Distribution (DOM, Henyey-Greenstein)')
plt.colorbar()
plt.show()



import numpy as np 
import matplotlib.pyplot as plt 
from scipy import interpolate

""""
def wavprop(F, xV, tV, Nphi, R0):
    Xi = -xV[0]
    Xo = 2 * Xi
    Nx = len(xV) # Number of spatial points per axis.
    Nt = len(tV) # Number of time steps.
    dx = xV[1] - xV[0]
    dt = tV[1] - tV[0]
    Nxo = 2 * Nx # The new, expanded spatial grid size (doubling to avoid aliasing in FFT).
    xVo = np.linspace(-Xo, Xo - 2*Xo / Nxo, Nxo) # The extended spatial grid, ranging from -Xo to Xo.

    # (Detector angles and coordinates) Detectors are evenly placed on a circle around the origin.
    phiV = np.linspace(0, (2 - 2 / Nphi) * np.pi, Nphi)
    XC = R0 * np.sin(phiV) # np.sqrt(2)
    YC = R0 * np.cos(phiV) # np.sqrt(2)

    # k-space grid (Constructs the 2D wavenumber grid for Fourier space).
    kV = np.linspace(-Nxo / 2, Nxo / 2 - 1, Nxo) * np.pi / Xi
    k1, k2 = np.meshgrid(kV, kV)
    km_sq = np.fft.ifftshift(k1**2 + k2**2) # ifftshift centers the zero-frequency component.
    km = np.sqrt(km_sq) # spatial frequency.

    # Zero-initialize pressure fields
    Data = np.zeros((Nt, Nphi)) # Stores the pressure values recorded by detectors.
    P = np.zeros((Nxo, Nxo))

    # Pad F to fit larger grid
    pad_width = (Nxo - F.shape[0]) // 2
    F_padded = np.pad(F, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant')

    # (F_padded) Initial pressure distribution padded to the larger grid size (Nxo x Nxo)

    # Initial wave field setup
    P += F_padded
    W = P.copy() # The current wavefields.
    Walt = W.copy() # The previous wavefields.
    V = W - P #Velocity-like term used in the finite difference step.

    # Time propagation loop
    for ii in range(Nt):
        # Convert wavefields into Fourier domain for efficient spatial derivative computation.
        FV = np.fft.fft2(V)
        FW = np.fft.fft2(W)

        # K-space time integration step
        # Applies k-space pseudo-spectral update rule, which is a stable, accurate way to advance wave equations.
        sin_term = np.sin(np.pi/8 * dt * km)**2
        LW = -4 * sin_term * (FW - FV)
        LW1 = np.fft.ifft2(LW)

        # Wave update (using a second-order accurate scheme).
        Wneu = 2 * W - Walt + LW1
        Walt = W.copy()
        W = Wneu

        # Interpolation on circular detector
        # Uses 2D spline interpolation to extract values of the pressure field at the detector positions.
        f = interpolate.RectBivariateSpline(xVo, xVo, np.real(W))
        Pcirc_time = f(XC, YC, grid=False)
        Data[ii, :] = Pcirc_time

        # Optional: animation (uncomment to enable)
        if ii % 10 == 0:
            plt.clf()
            plt.imshow(np.real(W), extent=[-Xo, Xo, -Xo, Xo], cmap='RdBu', vmin=-1.5, vmax=1.5)
            plt.plot(XC, YC, 'k.', markersize=2)
            plt.title(f"Wavefield at t = {ii * dt:.3f}")
            plt.pause(0.01)

    return Data
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import shift


def wavprop(F, xV, tV, Nphi, R0):
    # --- Set up space ---
    Xi = xV[-1]  # правый конец основной области
    Xo = 2 * Xi  # внешняя большая область (в 2 раза больше)
    Nx = len(xV)
    Nt = len(tV)
    dx = xV[1] - xV[0]
    dt = tV[1] - tV[0]
    Nxo = 2 * Nx  # увеличенная сетка
    # xVo = np.linspace(-Xo, Xo, Nxo)
    xVo = np.linspace(-Xo + dx/2, Xo - dx/2, Nxo)  # центрированная сетка


    # --- Set up detectors ---
    phiV = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    XC = R0 * np.sin(phiV)
    YC = R0 * np.cos(phiV)

    # --- Set up k-space grid ---
    kV = np.linspace(-Nxo/2, Nxo/2 - 1, Nxo) * np.pi / Xi
    k1, k2 = np.meshgrid(kV, kV)
    km_sq = np.fft.ifftshift(k1**2 + k2**2)
    km = np.sqrt(km_sq)

    # --- Initialize fields ---
    Data = np.zeros((Nt, Nphi))
    # P = np.zeros((Nxo, Nxo))


    pad = (Nxo - F.shape[0]) // 2
    F_padded = np.pad(F, ((pad, pad), (pad, pad)), mode='constant')
    """"
    # --- Pad initial field ---
    pad_diff = Nxo - F.shape[0]
    pad_before = pad_diff // 2
    pad_after = pad_diff - pad_before
    F_padded = np.pad(F, ((pad_before, pad_after), (pad_before, pad_after)), mode='constant')
    """
    """"
    P += F_padded
    W = P.copy()
    Walt = W.copy()
    V = W - P  # velocity-like field
    """
    W = np.zeros_like(F_padded)           # нулевая начальная деформация
    Walt = -dt * F_padded                 # инициализация по производной (начальная скорость)
    V = W - Walt                          # текущая скорость


    alpha = 0
    # It is not necessary if:
    # step by step contributes to the Courant condition: c⋅Δt<Δx.
    # the wave equation does not give instability.
    # Goal:
    # Maximum precision: 0
    # High frequency noise suppression: 0.001 or smaller.
    # Prevent numerical spikes: > 0.

    # --- Time propagation loop ---
    for ii in range(Nt):
        FV = np.fft.fft2(V)
        FW = np.fft.fft2(W)

        sin_term = np.sin(dt * km / 2)**2
        LW = -4 * sin_term * (FW - FV) - alpha * FW
        LW1 = np.fft.ifft2(LW)

        Wneu = 2 * W - Walt + LW1
        Walt = W.copy()
        W = Wneu
        V = W - Walt

        # Interpolate on detector circle
        f = interpolate.RectBivariateSpline(xVo, xVo, np.real(W))
        Pcirc_time = f(XC, YC, grid=False)
        Data[ii, :] = Pcirc_time


    """
    print(F.shape)           # Ожидается: (Nx, Nx)
    print(len(xV), len(tV))  # Nx, Nt
    print(F_padded.shape)    # Ожидается: (Nxo, Nxo)

    print("xVo min/max:", xVo.min(), xVo.max())
    print("XC range:", XC.min(), XC.max())
    print("YC range:", YC.min(), YC.max())

   # Visualization for debugging
    plt.figure()
    plt.plot(XC, YC, 'ro')
    plt.gca().set_aspect('equal')
    plt.title("Detectors")
    plt.grid()
    plt.show()

    plt.imshow(F_padded, cmap='gray')
    plt.title("Initial Pressure (F_padded)")
    plt.colorbar()
    plt.show()

    plt.plot(Data[:, 0])
    plt.title("Signal at One Detector")
    plt.xlabel("Time step")
    plt.ylabel("Pressure")
    plt.grid()
    plt.show()"""


    return Data

"""
# --- Convert light fluence to initial pressure ---
absorption_map = absorption_coeff_vector.reshape(Phi_total.shape)
#print("Phi_total.shape:", Phi_total.shape)
# print("absorption_coeff_vector.shape:", absorption_coeff_vector.shape)
F = Phi_total * absorption_map  # physically accurate

F = F / np.max(np.abs(F))  # нормализация к [0,1]


# --- Detector and time parameters ---
T = 3.0                  # total observation time
Nt = 451             # number of time steps
# xV_old  = np.linspace(-Lx, Lx , nx+1)
xV_old  = np.linspace(-Lx, Lx , nx)

xV = xV_old[: nx-1]
tV = np.linspace(0, T, Nt)
Nphi = 100               # number of detectors on the circle
R0 = 0.9 * np.sqrt(2) * Lx            # radius of the circle (slightly smaller than domain)
"""


plt.figure(figsize=(10, 4))
plt.imshow(abs_coeff_map,  origin='lower')
plt.colorbar(label='Pressure')
plt.title('Absorbtion coeff')
plt.tight_layout()
plt.show()

absorption_map = absorption_coeff_vector.reshape(Phi_total.shape)
F = Phi_total * absorption_map
F = F / np.max(np.abs(F))


plt.figure(figsize=(10, 4))
plt.imshow(Phi_total,  origin='lower')
plt.colorbar(label='Pressure')
plt.title('Total Phi')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.imshow(F,  origin='lower')
plt.colorbar(label='Pressure')
plt.title('Absorbed energy')
plt.tight_layout()
plt.show()



xV = np.linspace(-Lx / 2, Lx / 2, nx)
X, Y = np.meshgrid(xV, xV)

# --- Центрируем оптические данные ---
shift_x = nx // 2 - np.argmax(np.sum(Phi_total, axis=0))
shift_y = nx // 2 - np.argmax(np.sum(Phi_total, axis=1))
Phi_centered = shift(Phi_total, shift=(shift_y, shift_x), mode='nearest')
abs_centered = shift(absorption_map, shift=(shift_y, shift_x), mode='nearest')

F = Phi_centered * abs_centered
F = F / np.max(np.abs(F))

c = 1500.0  # м/с
dx = xV[1] - xV[0]
dt_max = 0.8 * dx / c
Nt = 800
R0 = 0.7 * np.sqrt(2) * (Lx / 2)
T = 2 * R0 / c * 1.2
tV = np.linspace(0, T, Nt)
Nphi = 200



# --- Simulate wave propagation ---
Data = wavprop(F, xV, tV, Nphi, R0)

# --- Visualize the signal ---
plt.figure(figsize=(10, 4))
plt.imshow(Data.T, aspect='auto', cmap='hot', extent=[0, T, 0, 360])
plt.colorbar(label='Pressure')
plt.xlabel('Time')
plt.ylabel('Angle (deg)')
plt.title('Photoacoustic Signals from DOM-Simulated Illumination')
plt.tight_layout()
plt.show()



from scipy.interpolate import interp1d

def kernel(tV):
    N = len(tV)
    Amat = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            Amat[i, j] = np.sqrt(tV[i]**2 - tV[j]**2) - np.sqrt(tV[i]**2 - tV[j+1]**2)
    return Amat


def wavstarprop(data, xV, tV, R0, T0):
    (Nt, Nphi) = data.shape
    Nx = len(xV)
    dt = tV[1] - tV[0]
    dphi = 2 * np.pi / Nphi
    phiV = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    X, Y = np.meshgrid(xV, xV)
    MF = np.zeros((Nphi, Nt))
    Amat = kernel(tV)
    F1 = np.zeros((Nx, Nx))

    # Предобработка сигналов
    for iicent in range(Nphi):
        p = data[:, iicent]
        # Градиент вместо дифференцирования
        p = np.gradient(p, dt)
        mf = np.matmul(Amat.T, p)
        tVmod = tV.copy()
        tVmod[0] = 0.5 * dt  # избежание деления на 0
        mf = mf / tVmod
        MF[iicent, :] = mf

    # Основной цикл реконструкции
    for icent in range(Nphi):
        phi = phiV[icent]
        xc = R0 * np.cos(phi)
        yc = R0 * np.sin(phi)
        mf = MF[icent, :]

        D = np.sqrt((X - xc)**2 + (Y - yc)**2)
        # interp_fn = interp1d(tV, mf, bounds_error=False, fill_value=0)
        # fadd = -interp_fn(D.ravel()).reshape(D.shape)
        fadd = -np.interp(D / c, tV, mf, left=0, right=0)


        F1 += fadd * dphi


    # Ограничение внутри круга
    RR = np.sqrt(X**2 + Y**2)
    WBP = F1.T
    WBP[RR > R0] = 0
    WBP = - WBP / np.max(np.abs(WBP))  # нормировка


    return WBP

from scipy.ndimage import shift as img_shift

recon = wavstarprop(Data, xV, tV, R0, T)

max_F = np.unravel_index(np.argmax(F), F.shape)
max_recon = np.unravel_index(np.argmax(recon), recon.shape)

shift_y = max_F[0] - max_recon[0]
shift_x = max_F[1] - max_recon[1]

print(f"Applying shift: ({shift_y}, {shift_x})")
recon_aligned = img_shift(recon, shift=(shift_y, shift_x), mode='nearest')

""""

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(F, vmin=0, vmax=1, cmap='hot', extent = [xV[0], xV[-1], xV[0], xV[-1]], origin='lower')
plt.title("True Initial Pressure")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(recon_aligned, vmin=0, vmax=1, cmap='hot', extent = [xV[0], xV[-1], xV[0], xV[-1]], origin='lower')
plt.title("Reconstructed Pressure")
plt.colorbar()

plt.tight_layout()
plt.show()

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utils as u

# parameters
G = 1.0
t, tEnd = 0., 5.
dt = 0.002
N = int(1e2)
softening = 0.1
totalMass = 5.
boxSize = 2.
meanVel = 0.
velDisp = 1
massDisp = 0.1

# setting random seed for reproducibility
np.random.seed(1)

# initializing particles
mass = totalMass * np.ones((N, 1))/N
pos = np.concatenate((np.random.normal(-0.5, massDisp, size=(int(N/2), 3)),
                     np.random.normal(0.5, massDisp, size=(int(N/2), 3))),
                     axis=0)  # this configuration is kind of like two bodies colliding
vel = np.random.normal(meanVel, velDisp, size=(N, 3))

# convert to CoM frame
vel -= np.mean(mass * vel, 0) / np.mean(mass)

# calculate initial conditions
accel = u.getAccel(pos, mass, G, softening)
KE, PE = u.getEnergy(pos, vel, mass, G)

# calculate number of timesteps
Nt = int(np.ceil(tEnd / dt))
tArr = np.arange(Nt + 1) * dt

# save values to be plotted
pos_save = np.zeros((N, 3, Nt + 1))
pos_save[:,:,0] = pos
spd_save = np.zeros((N, Nt + 1))
spd_save[:, 0] = np.linalg.norm(vel, axis=1)
KE_save = np.zeros(Nt + 1)
KE_save[0] = KE
PE_save = np.zeros(Nt + 1)
PE_save[0] = PE

# main loop
for i in range(Nt):
    if i % np.floor(Nt/10) == 0:
        print(f'{int(100*i/Nt)}% done')
    # update accelerations with kick-drift-kick stepping
    vel += accel * 0.5 * dt
    pos += vel * dt
    accel = u.getAccel(pos, mass, G, softening)
    vel += accel * 0.5 * dt
    t += dt

    # get energy
    KE, PE = u.getEnergy(pos, vel, mass, G)

    # save data
    pos_save[:,:,i+1] = pos
    KE_save[i+1] = KE
    PE_save[i+1] = PE
    spd_save[:,i+1] = np.linalg.norm(vel, axis=1)

# stacking energy data
energy_save = np.vstack((KE_save, PE_save, KE_save+PE_save))

# initialize figure
fig = plt.figure(figsize=(8,10), dpi=80)
gs = plt.GridSpec(3, 1, wspace=0., hspace=0.3)

ax1 = plt.subplot(gs[:2, 0])

ax1.set(title=f'N-body simulation with $N = {N}$, $M_{{tot}} = {totalMass}$',
        xlim=(0., boxSize), ylim=(0., boxSize),
        xticks=[], yticks=[],
        facecolor='k')
ax1.set_aspect('equal', 'box')

ax2 = plt.subplot(gs[2, 0])
ax2.set(xlim=(0, tEnd),
        xlabel='time (s)', ylabel='energy',
        facecolor='white')

# artists to be updated
posScatter = ax1.scatter([], [], s=5, c=[], vmin=0, vmax=np.max(spd_save), cmap='plasma')
it_text = ax1.text(0.02, 0.97, '',
                   horizontalalignment='left', verticalalignment='top',
                   transform=ax1.transAxes, color='white')
KELine = ax2.plot([], [], lw=.5, color='r', label='$KE$')[0]
PELine = ax2.plot([], [], lw=.5, color='b', label='$PE$')[0]
totalLine = ax2.plot([], [], lw=1., color='k', label='$E_{total}$')[0]

# decorating graphs
ax2.set(ylim=(1.5*np.min(energy_save), 1.5*np.max(energy_save)))
ax2.legend(loc='upper right')
fig.colorbar(posScatter, ax=ax1, label='speed')


# setup animator
def init():
    posScatter.set_offsets(np.c_[[], []])
    it_text.set_text('')
    KELine.set_data([], [])
    PELine.set_data([], [])
    totalLine.set_data([], [])
    return posScatter, it_text, KELine, PELine, totalLine


def animate(i):
    # animating particles
    x = pos_save[:, 0, i]
    y = pos_save[:, 1, i]
    spd = spd_save[:, i]
    posScatter.set_offsets(np.c_[x, y])
    posScatter.set_array(spd)

    # animating energy lines
    t = tArr[:i]
    KELine.set_data(t, KE_save[:i])
    PELine.set_data(t, PE_save[:i])
    totalLine.set_data(t, KE_save[:i] + PE_save[:i])

    # rescaling axes
    scale = 1.0
    std = (np.std(x) + np.std(y))/2
    x_len = scale * boxSize * std
    y_len = scale * boxSize * std
    ax1.set(xlim=(-x_len, x_len), ylim=(-y_len, y_len))

    # animating text
    it_text.set_text(f't: {round(i*dt,2)} s\n'
                     f'box width: {round(2*x_len,2)}')

    return posScatter, it_text, KELine, PELine, totalLine


# call animator
print('Animating..')
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt+1, interval=16.7, blit=False, repeat=True)

# showing plot if animation time is short
if N <= 1e2:
    plt.show()

# save animation
print('Saving animation..')
anim.save('nbody.mp4', fps=60, extra_args=['-vcodec', 'libx264'], dpi=80)
print('Saved')

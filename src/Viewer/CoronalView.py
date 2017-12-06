import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wid


def CoronalView(array):

    final_volume = array
    final_volume =np.swapaxes(final_volume, 2,0)
    final_volume = final_volume[::-1,:,:]
    vals = np.unique(final_volume)
    old_max = vals[-1]
    old_min = vals[0]
    new_max = 1.0
    new_min = -1.0

    final_volume = (final_volume-old_min) * (new_max-new_min) / (old_max-old_min) + new_min

    fig, ax = plt.subplots(1, 1)
    axcolor = 'lightgoldenrodyellow'
    axSlice = plt.axes([0.2, 0.05, 0.6, 0.03], axisbg=axcolor )
    sliceSlider = wid.Slider(axSlice, 'Slice', 0, final_volume.shape[1]-1, valinit=int((final_volume.shape[1]/2)))

    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')

            self.X = X
            self.rows, self.cols, self.slices = X.shape

            self.ind = self.cols//2

            self.im = ax.imshow(self.X[:, self.ind, :], cmap='gray')
            self.update()

        def on_scroll(self, event):
            if event.button == 'up':
                self.ind = np.clip(self.ind + 1, 0, self.cols - 1)
            else:
                self.ind = np.clip(self.ind - 1, 0, self.cols - 1)
            self.update()

        def update(self):
            self.im.set_data(self.X[:,self.ind, :])
            ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    def update(val):
        tracker.ind = int(sliceSlider.val)
        tracker.im.set_data(tracker.X[:, tracker.ind, :])
        ax.set_ylabel('slice %s' % tracker.ind)
        tracker.im.axes.figure.canvas.draw()

    tracker = IndexTracker(ax, final_volume)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    sliceSlider.on_changed(update)
    plt.show()

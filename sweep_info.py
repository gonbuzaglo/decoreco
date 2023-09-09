import torch
import analysis
import analysis_utils
from analysis import find_nearest_neighbour, scale, sort_by_metric
from settings import sweeps_dir


class SweepInfo():
    def __init__(self, id):
        self.id = id
        sweep = analysis_utils.read_sweep(sweeps_dir, self.id, name=None)
        self.problem = sweep.problem
        self.args, self.Xtrn, self.Ytrn, ds_mean, W, self.model = analysis_utils.sweep_get_data_model(sweep, put_in_sweep=True,
                                                                                            run_train_test=True)
        self.tst_error = sweep.tst_error
        self.tst_error = sweep.trn_error
        X = analysis_utils.get_all_reconstruction_outputs(sweep, verbose=True)
        # Find Nearest Neighbour
        if self.problem == 'mnist_odd_even' or self.problem == 'mnist_binary' or self.problem == 'mnist_multiclass':
            xx = find_nearest_neighbour(X, self.Xtrn, search='ncc', vote='min', use_bb=False, nn_threshold=None)
        else:
            xx = find_nearest_neighbour(X, self.Xtrn, search='ncc2', vote='mean', use_bb=True, nn_threshold=1.1)
        # Scale to Images
        xx, yy = scale(xx, self.Xtrn, ds_mean)
        # # Sort
        if self.problem == 'mnist_odd_even' or self.problem == 'mnist_binary' or self.problem == 'mnist_multiclass':
            self.xx, self.yy, self.ssims, self.sort_idxs = sort_by_metric(xx, yy, sort='l2')
        else:
            self.xx, self.yy, self.ssims, self.sort_idxs = sort_by_metric(xx, yy, sort='ssim')


        self.num_classes = self.args.num_classes
        self.values = self.model(self.Xtrn).data

    def show_table(self, fig_elms_in_line=20, fig_lines_per_page=10, fig_type='one_above_another', **kwargs):
        analysis.plot_table(self.xx, self.yy, fig_elms_in_line, fig_lines_per_page,
                            fig_type=fig_type, **kwargs)

    # for multiclass
    def _get_classes_plot_data(self):
        classes_plot_data = {}
        for c in range(self.num_classes):
            vals = self.values[self.Ytrn == c].clone()
            p = vals[:, c].clone()
            vals[:, c] = -torch.inf
            second_best = vals.max(dim=1)[0]
            margin = p - second_best
            margin_dist = list(zip(margin.data.cpu(), self.ssims[self.Ytrn == c].cpu()))
            classes_plot_data[c] = sorted(margin_dist, key=lambda x: x[0])
        return classes_plot_data

    # for multiclass
    def _plot_c_dpc(self, ax, header=''):
        classes_plot_data = self._get_classes_plot_data()
        for c in range(self.num_classes):
            xy = classes_plot_data[c]
            x = [i[0] for i in xy]
            y = [i[1] for i in xy]
            ax.scatter(x, y, marker='o', s=10)
        ax.grid('both')
        ax.set_xlabel('$\phi_{\\theta}(x)$')
        ax.set_ylabel('SSIM$(x, \hat{x})$')
        ax.axhline(y=0.4, color='k', linestyle='-')
        ax.set_title(header, fontsize=10)

    # for binary
    def _get_plot_data(self):
        output_ssim = list(zip(self.values.data.cpu(), self.ssims.cpu()))
        plot_data = sorted(output_ssim, key=lambda x: x[0])
        return plot_data

    # for binary
    def plot_ssim_to_outputs(self, ax, header=''):
        xy = self._get_plot_data()
        x = [i[0] for i in xy]
        y = [i[1] for i in xy]
        ax.scatter(x, y, marker='o', s=10)
        ax.grid('both')
        ax.set_xlabel('$\phi_{\\theta}(x)$')
        ax.set_ylabel('SSIM$(x, \hat{x})$')
        ax.axhline(y=0.4, color='grey', linestyle='--', lw=1)
        ax.set_title(header, fontsize=10)

    # for regression
    def _get_plot_data_for_regression(self, absolute=True):
        values = self.values.data.view(-1).cpu()
        error = values - self.Ytrn.data.cpu()
        if absolute:
            error = abs(error)
        output_ssim = list(zip(error, self.ssims.cpu()))
        plot_data = sorted(output_ssim, key=lambda x: x[0])
        return plot_data

    # for regression
    def plot_ssim_to_error_regression(self, ax, header='', absolute=True):
        xy = self._get_plot_data_for_regression(absolute)
        x = [i[0] for i in xy]
        y = [i[1] for i in xy]
        ax.scatter(x, y, marker='o', s=10)
        ax.grid('both')
        if absolute:
            ax.set_xlabel('$|\phi_{\\theta}(x)-y(x)|$')
        else:
            ax.set_xlabel('$\phi_{\\theta}(x)-y(x)$')
        ax.set_ylabel('SSIM$(x, \hat{x})$')
        ax.axhline(y=0.4, color='grey', linestyle='--', lw=1)
        ax.set_title(header, fontsize=10)

    def _get_plot_data_for_l_tag(self, loss_fn):
        phi = self.model(self.Xtrn).squeeze().data
        phi = phi.requires_grad_(True)
        loss_fn(phi, self.Ytrn.data).backward()
        output_ssim = list(zip(phi.grad.abs().cpu(), self.ssims.cpu()))
        plot_data = sorted(output_ssim, key=lambda x: x[0])
        return plot_data

    # for regression
    def plot_ssim_to_l_tag(self, ax, loss_fn, header=''):
        xy = self._get_plot_data_for_l_tag(loss_fn)
        x = [i[0] for i in xy]
        y = [i[1] for i in xy]
        ax.scatter(x, y, marker='o', s=10)
        ax.grid('both')
        ax.set_xlabel('$|\\frac{\partial\mathcal{L}}{\partial\phi_{\\theta}(x)}|$')
        ax.set_ylabel('SSIM$(x, \hat{x})$')
        ax.axhline(y=0.4, color='grey', linestyle='--', lw=1)
        ax.set_title(header, fontsize=10)


    def _get_plot_data_for_loss(self, loss_fn):
        phi = self.model(self.Xtrn).squeeze().data
        phi = phi.requires_grad_(True)
        loss = loss_fn(phi, self.Ytrn.data)
        output_ssim = list(zip(loss.cpu(), self.ssims.cpu()))
        plot_data = sorted(output_ssim, key=lambda x: x[0])
        return plot_data

    # for regression
    def plot_ssim_to_loss(self, ax, loss_fn, header=''):
        xy = self._get_plot_data_for_loss(loss_fn)
        x = [i[0] for i in xy]
        y = [i[1] for i in xy]
        ax.scatter(x, y, marker='o', s=10)
        ax.grid('both')
        ax.set_xlabel('$\mathcal{L}$')
        ax.set_ylabel('SSIM$(x, \hat{x})$')
        ax.axhline(y=0.4, color='grey', linestyle='--', lw=1)
        ax.set_title(header, fontsize=10)

    # make it look like multiclass
    def _get_classes_plot_data_for_binary(self):
        margin_dist = list(zip(self.values.data.cpu(), self.ssims.cpu()))
        classes_plot_data = sorted(margin_dist, key=lambda x: x[0])
        return classes_plot_data

    # make it look like multiclass
    def _plot_c_dpc_for_binary(self, ax, header=''):
        classes_plot_data = self._get_classes_plot_data_for_binary()
        xy = classes_plot_data
        x_neg, y_neg, x_pos, y_pos = [], [], [], []
        for i in xy:
            if i[0] < 0:  # vehicle
                x_neg.append(abs(i[0]))
                y_neg.append(i[1])
            else:
                x_pos.append(i[0])
                y_pos.append(i[1])
        ax.scatter(x_neg, y_neg, marker='o', s=10, c='red')  # red for vehicles
        ax.scatter(x_pos, y_pos, marker='o', s=10, c='green')  # green for animals
        ax.grid('both')
        ax.set_xlabel('$\phi_{\\theta}(x)$')
        ax.set_ylabel('SSIM$(x, \hat{x})$')
        ax.axhline(y=0.4, color='k', linestyle='-')
        ax.set_title(header, fontsize=10)

    def plot_c_dpc(self, ax, header=''):
        if self.problem == 'cifar10_vehicles_animals':
            self._plot_c_dpc_for_binary(ax, header)
        else:
            self._plot_c_dpc(ax, header)

    def count_good(self, threshold=0.4):
        return torch.numel(self.ssims[self.ssims > threshold])
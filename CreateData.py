
def setup_problem(args):
    if False:
        pass
    elif args.problem == 'cifar10_vehicles_animals':
        from problems.cifar10_vehicles_animals import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'cifar10_vehicles_animals_regression':
        from problems.cifar10_vehicles_animals_regression import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'cifar10_multiclass':
        from problems.cifar10_multiclass import get_dataloader
        return get_dataloader(args)
    elif args.problem == 'cifar100_multiclass':
        from problems.cifar100_multiclass import get_dataloader
        return get_dataloader(args)
    else:
        raise ValueError(f'Unknown args.problem={args.problem}')
    return data_loader


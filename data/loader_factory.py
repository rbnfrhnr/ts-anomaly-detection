from data.ucr_loader import load as ucr_loader


def get_loader(data_set):
    if data_set == 'ucr':
        return ucr_loader

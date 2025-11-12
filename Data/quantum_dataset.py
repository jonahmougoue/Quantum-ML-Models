import os
import pathlib
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import requests
import zipfile

class QuantumDataset(Dataset):
    def __init__(self, potentials:str='all'):
        '''
        Dataset containing 25000 items, each with:\n
        -1x256x256 potential map\n
        -1x256x356 wavefunction^2 map\n
        -Energy of the wavefunction\n
        -Label for the potential\n
        Data from https://nrc-digital-repository.canada.ca/eng/view/object/?id=1343ae23-cebf-45c6-94c3-ddebdb2f23c6
        :param potentials: Name of potential to use. 'all' selects all potentials in the dataset.
        '''
        potentials = potentials.lower()

        arg_to_file = {
            'harmonic oscillator': 'HO_gen2_0010.h5',
            'infinite well': 'IW_gen2_0010.h5',
            'negative gaussian': 'NG_gen2b_0000.h5',
            'random': 'RND_0011.h5',
            'random ke': 'RND_KE_gen2_0010.h5',
        }

        data_folder = (pathlib.Path(__file__).parent / pathlib.Path('')).resolve()
        sample_folder = pathlib.Path('SAMPLE')
        zip_folder = 'SAMPLE.zip'
        url = 'https://nrc-digital-repository.canada.ca/eng/view/sample/?id=1343ae23-cebf-45c6-94c3-ddebdb2f23c6'

        if not os.path.exists(data_folder / sample_folder):
            tqdm.write('Dataset not downloaded\n')

            with requests.get(url,stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 1024*64*64
                if not os.path.exists(data_folder / zip_folder):
                    with open(data_folder / zip_folder, 'wb') as f:
                        for data in tqdm(r.iter_content(block_size),
                                         total=total_size//block_size,
                                         desc='Downloading Files',
                                         unit='blocks',
                                         leave=False):
                            f.write(data)
                        tqdm.write(f'Downloaded {zip_folder} to {data_folder}\n')
                with zipfile.ZipFile(data_folder / zip_folder, 'r') as f:
                    for item in tqdm(f.infolist(),
                                     desc='Extracting Files',
                                     unit='files',
                                     leave=False):
                        f.extract(item, data_folder)
                    tqdm.write(f'Extracted to {data_folder}\n')

        if potentials == 'all':
            self.files = os.listdir(data_folder / sample_folder)
        else:
            self.files = [arg_to_file[potentials]]

        for file in self.files:
            with h5py.File(data_folder / sample_folder / file, 'r') as f:
                print(file)
                for col in f:
                    print(f'{col}: {f[col].shape}')
            print('')

        calculated_energy = []
        wavefunction = []
        potential = []
        potential_label = []

        for file_id, file in tqdm(enumerate(self.files),total=len(self.files)):
            with h5py.File(data_folder / sample_folder / file, 'r') as f:

                wavefunction.append(f['wavefunction'][:]) if 'wavefunction' in f else wavefunction.append(f['psi'][:])
                calculated_energy.append(f['calculated_energy'][:])
                potential.append(f['potential'][:])
                potential_label.append([file_id]*len(f['potential'][:]))

        self.calculated_energy = torch.from_numpy(np.concatenate(calculated_energy))
        self.potential = torch.from_numpy(np.concatenate(potential))
        self.wavefunction = torch.from_numpy(np.concatenate(wavefunction))
        self.potential_label = torch.from_numpy(np.concatenate(potential_label))

    def __len__(self):
        return len(self.calculated_energy)

    def __getitem__(self,idx):
        return {
            'potential': self.potential[idx],
            'wavefunction2': self.wavefunction[idx],
            'energy': self.calculated_energy[idx],
            'potential_label': self.potential_label[idx],
        }

    def get_files(self):
        '''
        Gets the names of each file being used
        :return: List of file names
        '''
        return self.files

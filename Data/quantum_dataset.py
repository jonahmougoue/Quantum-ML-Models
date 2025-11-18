import os
import pathlib
import h5py
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import requests
import zipfile



class QuantumDataset(Dataset):
    def __init__(self, potentials:str='all',memory=False):
        """
        Dataset containing 25,000 samples, each with:\n
        -1x256x256 potential map\n
        -1x256x256 Ground state wavefunction^2 map\n
        -Ground state energy\n
        -Label identifying potential type\n
        Data sourced from: https://nrc-digital-repository.canada.ca/eng/view/object/?id=1343ae23-cebf-45c6-94c3-ddebdb2f23c6
        :param potentials: Name of potential to use. Options include
            ['harmonic oscillator','infinite well','negative gaussian','random','random ke','all']
        :param memory: If True, loads data on memory.
        """
        self.memory = memory

        potentials = potentials.lower()

        label_to_file = {
            'harmonic oscillator': 'HO_gen2_0010.h5',
            'infinite well': 'IW_gen2_0010.h5',
            'negative gaussian': 'NG_gen2b_0000.h5',
            'random': 'RND_0011.h5',
            'random ke': 'RND_KE_gen2_0010.h5',
        }
        file_to_label = {
            'HO_gen2_0010.h5':'Harmonic Oscillator',
            'IW_gen2_0010.h5':'Infinite Well',
            'NG_gen2b_0000.h5':'Negative Gaussian',
            'RND_0011.h5':'Random Potential',
            'RND_KE_gen2_0010.h5':'Random Potential with Kinetic Energy',
        }

        data_folder = (pathlib.Path(__file__).parent / pathlib.Path('')).resolve()
        self.sample_folder = pathlib.Path('SAMPLE')
        zip_folder = 'SAMPLE.zip'
        url = 'https://nrc-digital-repository.canada.ca/eng/view/sample/?id=1343ae23-cebf-45c6-94c3-ddebdb2f23c6'

        if not os.path.exists(data_folder / self.sample_folder):
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
            self.files = sorted(os.listdir(data_folder / self.sample_folder))
        else:
            self.files = [label_to_file[potentials]]

        self.potential_labels = {}
        for i, file in enumerate(self.files):
            self.potential_labels[i] = file_to_label[file]

        '''
        for file in self.files:
            with h5py.File(data_folder / self.sample_folder / file, 'r') as f:
                print(file)
                for col in f:
                    print(f'{col}: {f[col].shape}')
            print('')
        '''
        self.index_map = []
        for file_id, file in enumerate(tqdm(self.files, desc="Indexing")):
            with h5py.File(data_folder / self.sample_folder / file, 'r') as f:
                length = len(f['potential'])
                self.index_map += [(file_id, i) for i in range(length)]

        if self.memory:
            tqdm.write("Loading data into memory")
            all_potentials, all_wavefunctions, all_energies, all_labels = [], [], [], []

            for file_id, file in enumerate(tqdm(self.files, desc="Loading files to memory")):
                with h5py.File(data_folder / self.sample_folder / file, 'r') as f:
                    potentials = torch.from_numpy(f['potential'][:]).float()
                    wavefunctions = torch.from_numpy(
                        f['wavefunction'][:] if 'wavefunction' in f else f['psi'][:]
                    ).float()
                    energies = torch.from_numpy(f['calculated_energy'][:]).float()
                    labels = torch.full((len(potentials),), file_id, dtype=torch.long)

                all_potentials.append(potentials)
                all_wavefunctions.append(wavefunctions)
                all_energies.append(energies)
                all_labels.append(labels)

            self.potential = torch.cat(all_potentials)
            self.wavefunction2 = torch.cat(all_wavefunctions)
            self.energy = torch.cat(all_energies)
            self.potential_label = torch.cat(all_labels)

            del all_potentials, all_wavefunctions, all_energies, all_labels

        self.data_folder = data_folder

    def __len__(self):
        """
        Returns the total numbers of samples of the dataset
        :return: Length of dataset
        """
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        Returns single sample from the dataset.
        :param idx:  Index of the sample
        :return: Dict containing properties of the sample
        """
        if self.memory:
            return {'potential': self.potential[idx],
                    'wavefunction2': self.wavefunction2[idx],
                    'energy': self.energy[idx],
                    'potential_index': self.potential_label[idx],
            }

        file_id, local_idx = self.index_map[idx]
        file_path = self.data_folder /self.sample_folder/ self.files[file_id]

        with h5py.File(file_path, 'r') as f:
            potential = torch.from_numpy(f['potential'][local_idx])
            if 'wavefunction' in f.keys():
                wavefunction2 = torch.from_numpy(f['wavefunction'][local_idx])
            else:
                wavefunction2 = torch.from_numpy(f['psi'][local_idx])
            energy = torch.tensor(f['calculated_energy'][local_idx])

        return {'potential': potential,
                'wavefunction2': wavefunction2,
                'energy': energy,
                'potential_index': file_id,
        }

    def get_labels(self):
        """
        Gets the names of each file being used
        :return: Dict of potential labels
        """

        return self.potential_labels

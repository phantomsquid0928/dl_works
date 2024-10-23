import os
import cv2
import numpy as np

class GlaucomaDataset:
    def __init__(self, root_dirs, split='train', output_size=(256, 256)):
        self.output_size = output_size
        self.split = split
        self.images = []
        self.segs = []

        # Load data from directories
        for direct in root_dirs:
            self.image_filenames = []
            for path in os.listdir(os.path.join(direct, "Images_Square")):
                if not path.startswith('.'):
                    self.image_filenames.append(path)

            # Load images and corresponding segmentation masks
            for k in range(len(self.image_filenames)):
                print(f'Loading {split} image {k + 1}/{len(self.image_filenames)}...', end='\r')

                # Load image using cv2 and convert to RGB
                img_name = os.path.join(direct, "Images_Square", self.image_filenames[k])
                img = cv2.imread(img_name)  # Read image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, self.output_size)  # Resize image
                img = img.astype(np.float32) / 255.0  # Normalize image to [0, 1]
                
                img = np.transpose(img, (2, 0, 1))
                self.images.append(img)

                # If not in test split, load segmentation masks
                if split != 'test':
                    seg_name = os.path.join(direct, "Masks_Square", self.image_filenames[k][:-3] + "png")
                    mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
                    od = (mask == 1).astype(np.float32)  # Optic disc mask
                    oc = (mask == 2).astype(np.float32)  # Optic cup mask
                    od = cv2.resize(od, self.output_size, interpolation=cv2.INTER_NEAREST)
                    oc = cv2.resize(oc, self.output_size, interpolation=cv2.INTER_NEAREST)
                    self.segs.append(np.stack([od, oc], axis=0))  # Stack the masks

            print(f'Successfully loaded {split} dataset.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.split == 'test':
            return img
        else:
            seg = self.segs[idx]
            return img, seg
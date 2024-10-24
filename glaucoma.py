import os
import cv2
import numpy as np

class GlaucomaDataset:
    def __init__(self, root_dirs, split='train', output_size=(256, 256), vCDR_threshold=0.6):
        self.output_size = output_size
        self.split = split
        self.images = []
        self.labels = []
        self.vCDR_threshold = vCDR_threshold

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
                self.images.append(img)

                if split != 'test':
                    # Load segmentation masks
                    seg_name = os.path.join(direct, "Masks_Square", self.image_filenames[k][:-3] + "png")
                    mask = cv2.imread(seg_name, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
                    od = (mask == 1).astype(np.float32)  # Optic disc mask
                    oc = (mask == 2).astype(np.float32)  # Optic cup mask
                    od = cv2.resize(od, self.output_size, interpolation=cv2.INTER_NEAREST)
                    oc = cv2.resize(oc, self.output_size, interpolation=cv2.INTER_NEAREST)

                    # Calculate vCDR (Vertical Cup-to-Disc Ratio)
                    vCDR = self.calculate_vCDR(od, oc)

                    # Assign binary label (1 if glaucoma, 0 if not)
                    label = 1 if vCDR > self.vCDR_threshold else 0
                    self.labels.append(label)

            print(f'Successfully loaded {split} dataset.')

    def calculate_vCDR(self, od, oc):
        """Calculate the vertical cup-to-disc ratio (vCDR) from the optic disc (od) and optic cup (oc) masks."""
        # Find the vertical diameter (sum along the y-axis)
        od_vertical_diameter = np.sum(od, axis=0).max()
        oc_vertical_diameter = np.sum(oc, axis=0).max()

        # Calculate the vertical cup-to-disc ratio
        vCDR = oc_vertical_diameter / (od_vertical_diameter + 1e-7)  # Add a small value to avoid division by zero
        return vCDR

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx] if self.split != 'test' else None
        return img, label

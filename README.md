## Instructions

### Environment
- `pip install -r requirements.txt`

#### T2.1
- Model choice: ResNet18 trained on train.part1 with BCE loss
- `cd <directory/with/train.py>`
- Train
- - `python train.py --train_dataset_path <path/to/train1/train> --val_dataset_path <path/to/val/val>`
- Test
- - `python test.py --dataset_path <path/to/val/val>`
- - Metrics achieved on dataset train.part2: Accuracy = 0.99, Precision = 0.99, Recall = 0.99

#### T2.2
- Model choice: custom autoencoder trained on train.part1 with L1 loss
- `cd <directory/with/train.py>`
- Train
- - `python train.py --train_dataset_path <path/to/train1/train> --val_dataset_path <path/to/val/val>`
- Test
- - `python test.py --dataset_path <path/to/val/val>`
- - Metrics achieved on dataset train.part2: MSE = 0.236

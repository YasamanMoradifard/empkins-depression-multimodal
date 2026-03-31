# All Available Hyperparameters for Tuning

This document lists all hyperparameters that can be tuned in `MMFformer_finetuning.py`, corresponding to the arguments in `main.py` (lines 81-157).

## Training Hyperparameters

- **learning_rate** (`-lr`): Learning rate for optimizer
  - Example: `[1e-5, 5e-4, 1e-3]`
  
- **batch_size** (`-bs`): Batch size for training
  - Example: `[8, 16, 32]`
  
- **epochs** (`-e`): Number of training epochs
  - Example: `[30, 35, 40]`
  
- **optimizer** (`--optimizer`): Optimizer type
  - Options: `["Adam", "AdamW", "SGD", "RMSprop"]`
  
- **weight_decay** (`--weight_decay`): L2 regularization weight
  - Example: `[1e-4, 1e-3, 1e-2]`
  
- **lr_scheduler** (`--lr_scheduler`): Learning rate scheduler
  - Options: `["cos", "StepLR", "Plateau"]`
  
- **lr_patience** (`--lr_patience`): Patience for ReduceLROnPlateau scheduler
  - Example: `[5, 10, 15]`
  
- **lr_steps** (`--lr_steps`): Epochs to decay learning rate (for StepLR)
  - Example: `[[100, 200], [200, 400]]` (list of lists)
  
- **begin_epoch** (`--begin_epoch`): Starting epoch (usually 1)
  - Example: `[1]`

## Optimizer-Specific Hyperparameters

- **amsgrad** (`--amsgrad`): Use AMSGrad variant for Adam/AdamW
  - Options: `[0, 1]` (0 = False, 1 = True)
  
- **momentum** (`--momentum`): Momentum factor for SGD
  - Example: `[0.8, 0.9, 0.95]`
  
- **dampening** (`--dampening`): Dampening for SGD
  - Example: `[0.8, 0.9, 0.95]`

## Dropout Hyperparameters (Regularization)

- **fusion_dropout** (`--fusion_dropout`): Dropout rate for fusion layer
  - Example: `[0.3, 0.5, 0.7]`
  
- **audio_dropout** (`--audio_dropout`): Dropout rate for audio features
  - Example: `[0.3, 0.5, 0.7]`
  
- **visual_dropout** (`--visual_dropout`): Dropout rate for visual features
  - Example: `[0.3, 0.5, 0.7]`
  
- **classifier_dropout** (`--classifier_dropout`): Dropout before final classifier
  - Example: `[0.0, 0.1, 0.2]`
  
- **attention_dropout** (`--attention_dropout`): Dropout in attention mechanisms
  - Example: `[0.0, 0.1, 0.2]`
  
- **transformer_dropout** (`--transformer_dropout`): Dropout in transformer layers
  - Example: `[0.0, 0.1, 0.2]`

## Loss Function Hyperparameters (Regularization)

- **lambda_reg** (`--lambda_reg`): L2 regularization coefficient in loss
  - Example: `[1e-6, 1e-5, 1e-4]`
  
- **focal_weight** (`--focal_weight`): Weight for focal loss component
  - Example: `[0.3, 0.5, 0.7]`
  
- **l2_weight** (`--l2_weight`): Weight for L2 regularization in loss
  - Example: `[0.3, 0.5, 0.7]`
  
- **label_smoothing** (`--label_smoothing`): Label smoothing epsilon
  - Example: `[0.0, 0.1, 0.2, 0.3]`

## Early Stopping Hyperparameters

- **early_stopping_patience** (`--early_stopping_patience`): Number of epochs to wait before stopping
  - Example: `[3, 5, 7]`
  
- **early_stopping_delta** (`--early_stopping_delta`): Minimum improvement to reset patience
  - Example: `[0.0, 0.001, 0.01]`

## Usage Examples

### Example 1: Tune learning rate and batch size
```json
{
  "learning_rate": [1e-5, 5e-4, 1e-3],
  "batch_size": [16, 32]
}
```

### Example 2: Tune dropout rates
```json
{
  "learning_rate": [5e-4],
  "fusion_dropout": [0.3, 0.5, 0.7],
  "visual_dropout": [0.3, 0.5, 0.7],
  "attention_dropout": [0.0, 0.1, 0.2]
}
```

### Example 3: Tune optimizer and scheduler
```json
{
  "learning_rate": [5e-4],
  "optimizer": ["Adam", "AdamW", "SGD"],
  "lr_scheduler": ["cos", "StepLR", "Plateau"],
  "weight_decay": [1e-4, 1e-3]
}
```

### Example 4: Comprehensive tuning
```json
{
  "learning_rate": [1e-5, 5e-4],
  "batch_size": [16, 32],
  "weight_decay": [1e-4, 1e-3],
  "fusion_dropout": [0.3, 0.5],
  "label_smoothing": [0.0, 0.1],
  "early_stopping_patience": [5, 7]
}
```

## Notes

- The script performs **grid search**, testing all combinations of the specified hyperparameters
- Be careful with the number of combinations - it grows exponentially!
- Example: 3 learning rates × 3 batch sizes × 2 optimizers = 18 combinations
- Use `--test_one` flag to test just the first combination for debugging
- Best hyperparameters are automatically saved to `best_hyperparameters.json`


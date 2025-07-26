# FastFlow and experiment constants

BACKBONE_DEIT = "deit_base_distilled_patch16_224"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

BATCH_SIZE = 4
NUM_EPOCHS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Class-specific hyperparameters (based on observed performance)
# Updated for stability: smaller flow_steps (2–4), higher weight_decay (1e-4)
CLASS_SPECIFIC_PARAMS = {
    "metal_plate": {
        "lr": 1e-2,
        "num_epochs": 6,
        "weight_decay": 1e-4,  # Increased for stability
        "flow_steps": 2,       # Reduced for stability
        "hidden_ratio": 1.0,
    },
    "bracket_white": {
        "lr": 1e-3,
        "num_epochs": 8,
        "weight_decay": 1e-4,
        "flow_steps": 2,
        "hidden_ratio": 0.8,
    },
    "bracket_brown": {
        "lr": 1e-3,
        "num_epochs": 10,
        "weight_decay": 1e-4,
        "flow_steps": 2,
        "hidden_ratio": 0.8,
    },
    "bracket_black": {
        "lr": 1e-3,
        "num_epochs": 25,
        "weight_decay": 1e-4,
        "flow_steps": 4,
        "hidden_ratio": 1,
    },
    "default": {
        "lr": 1e-3,
        "num_epochs": 50,
        "weight_decay": 1e-3,
        "flow_steps": 2,
    }
}

# WideResNet50-specific hyperparameters (more powerful backbone)
# Updated for stability: smaller flow_steps (2–4), higher weight_decay (1e-4)
CLASS_SPECIFIC_PARAMS_WRESNET50 = {
    "metal_plate": {
        "lr": 1e-3,
        "num_epochs": 30,
        "weight_decay": 1e-3,
        "flow_steps": 2,
        "hidden_ratio": 0.8,
    },
    "bracket_white": {
        "lr": 1e-3,
        "num_epochs": 40,
        "weight_decay": 1e-3,
        "flow_steps": 2,
        "hidden_ratio": 0.8,
    },
    "bracket_brown": {
        "lr": 1e-4,
        "num_epochs": 40,
        "weight_decay": 1e-3,
        "flow_steps": 2,
        "hidden_ratio": 1.2,
    },
    "bracket_black": {
        "lr": 1e-4,
        "num_epochs": 10,
        "weight_decay": 1e-3,
        "flow_steps": 2,
        "hidden_ratio": 1.2,
    },
    "default": {
        "lr": 1e-3,
        "num_epochs": 30,
        "weight_decay": 1e-3,
        "flow_steps": 2,
        "hidden_ratio": 0.8,
    }
}

LOG_INTERVAL = 10
EVAL_INTERVAL = 10
CHECKPOINT_INTERVAL = 10 
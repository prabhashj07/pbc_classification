from torch import nn
from timm.models._factory import create_model
from .BloodNetViT import ViT

# Define the model mapping for timm models 
model_mapping = {
    "vit_base_patch16_224":{
        "family": "vit"
        },
    "vit_small_patch16_224": {
        "family": "vit"
        },
    "vit_base_patch32_224": {
        "family": "vit"
        },
    "vit_large_patch16_224": {
        "family": "vit"
        },
    "vit_large_patch32_224": {
        "family": "vit"
        },
    "vit_tiny_patch16_224": {
        "family": "vit"
        },
    "BloodNetViT":{
        "family": "vit"
        },
    # Add more models as needed with their respective configurations.
}

class Model(nn.Module):
    """Model definition using timm."""

    def __init__(self, model_name: str, num_classes: int):
        """
        Initialize Model instance.

        Args:
            model_name (str): Name of the model architecture.
            num_classes (int): Number of output classes.
        """
        super(Model, self).__init__()

        if model_name not in model_mapping:
            valid_options = ", ".join(model_mapping.keys())
            raise ValueError(
                f"Invalid model name: '{model_name}'. Available options: {valid_options}"
            )

        # If BloodNetViT is selected, use it specifically, otherwise use timm model
        if model_name == "BloodNetViT":
            self.model = ViT(num_classes=num_classes)  
        else:
            self.model = create_model(
                model_name, pretrained=True, num_classes=num_classes
            )

        # Freeze model parameters except the classifier head
        for param in self.model.parameters():
            param.requires_grad = False

        # Allow training of the classifier 
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "fc"):
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)

class ModelFactory:
    """
    Factory for creating different models based on their names.

    Args:
        name (str): The name of the model factory.
        num_classes (int): The number of output classes.

    Raises:
        ValueError: If the specified model factory is not implemented.
    """

    def __init__(self, name: str, num_classes: int):
        """
        Initialize ModelFactory instance.

        Args:
            name (str): The name of the model.
            num_classes (int): The number of output classes.
        """
        self.name = name
        self.num_classes = num_classes

    def __call__(self):
        """
        Create a model instance based on the provided name.

        Returns:
            Model: An instance of the selected model.
        """
        if self.name not in model_mapping:
            valid_options = ", ".join(model_mapping.keys())
            raise ValueError(
                f"Invalid model name: '{self.name}'. Available options: {valid_options}"
            )

        return Model(self.name, self.num_classes)


if __name__ == "__main__":
    model = ModelFactory("vit_base_patch16_224", 8)()
    print(model)

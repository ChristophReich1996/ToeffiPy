import autograd
from autograd import Tensor
from module import Module
import functional


class Loss(Module):

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Constructor method
        :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
        """
        # Call super constructor
        super(Loss, self).__init__()
        # Check parameter
        assert reduction in ['mean', 'sum', 'none'], 'Reduction {} is not available. Use `mean`, `sum` or `none`'
        # Save parameter
        self.reduction = reduction


class L1Loss(Loss):
    """
    This class implements the L1 loss as a module
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Constructor method
        :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
        """
        # Call super constructor
        super(L1Loss, self).__init__(reduction)

    def forward(self, prediction: Tensor, label: Tensor) -> Tensor:
        """
        Forward pass computes loss
        :param prediction: (Tensor) Prediction tensor
        :param label: (Tensor) One hot encoded label tensor
        :return: (Tensor) Loss value
        """
        # Compute loss
        return functional.l1_loss(prediction=prediction, label=label, reduction=self.reduction)


class MSELoss(Loss):
    """
    This class implements the mean squared error loss as a module
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Constructor method
        :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
        """
        # Call super constructor
        super(MSELoss, self).__init__(reduction)

    def forward(self, prediction: Tensor, label: Tensor) -> Tensor:
        """
        Forward pass computes loss
        :param prediction: (Tensor) Prediction tensor
        :param label: (Tensor) One hot encoded label tensor
        :return: (Tensor) Loss value
        """
        # Compute loss
        return functional.mse_loss(prediction=prediction, label=label, reduction=self.reduction)


class BinaryCrossEntropyLoss(Loss):
    """
    This class implements the binary class cross entropy loss
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Constructor method
        :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
        """
        # Call super constructor
        super(BinaryCrossEntropyLoss, self).__init__(reduction)

    def forward(self, prediction: Tensor, label: Tensor) -> Tensor:
        """
        Forward pass computes loss
        :param prediction: (Tensor) Prediction tensor
        :param label: (Tensor) One hot encoded label tensor
        :return: (Tensor) Loss value
        """
        # Compute loss
        return functional.binary_cross_entropy_loss(prediction=prediction, label=label, reduction=self.reduction)


class CrossEntropyLoss(Loss):
    """
    This class implements the multi class cross entropy loss
    """

    def __init__(self, reduction: str = 'mean') -> None:
        """
        Constructor method
        :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
        """
        # Call super constructor
        super(CrossEntropyLoss, self).__init__(reduction)

    def forward(self, prediction: Tensor, label: Tensor) -> Tensor:
        """
        Forward pass computes loss
        :param prediction: (Tensor) Prediction tensor
        :param label: (Tensor) One hot encoded label tensor
        :return: (Tensor) Loss value
        """
        # Compute loss
        return functional.cross_entropy_loss(prediction=prediction, label=label, reduction=self.reduction)


class SoftmaxCrossEntropyLoss(Loss):
    """
    This class implements the multi class softmax cross entropy loss. This module should be used over softmax + CEL
    because of better numerical stability.
    """

    def __init__(self, reduction: str = 'mean', axis: int = 1) -> None:
        """
        Constructor method
        :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
        :param axis: (int) Axis to apply softmax
        """
        # Call super constructor
        super(SoftmaxCrossEntropyLoss, self).__init__(reduction)
        # Save parameter
        self.axis = axis

    def forward(self, prediction: Tensor, label: Tensor) -> Tensor:
        """
        Forward pass computes loss
        :param prediction: (Tensor) Prediction tensor
        :param label: (Tensor) One hot encoded label tensor
        :return: (Tensor) Loss value
        """
        # Compute loss
        return functional.softmax_cross_entropy_loss(prediction=prediction, label=label, reduction=self.reduction,
                                                     axis=self.axis)

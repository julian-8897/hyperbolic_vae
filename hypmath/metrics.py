import geoopt
import torch

ball = geoopt.PoincareBall()


def PoincareDistance(X1, X2):
    """Calculates the poincare distance between two points on the poincare ball,
        used for calculation of the Silhouette score.
    """
    return ball.dist(torch.Tensor(X1), torch.Tensor(X2))

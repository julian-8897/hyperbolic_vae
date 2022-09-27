import torch.nn
import geoopt
import geoopt.manifolds.stereographic.math as pmath

ball = geoopt.PoincareBall(c=1.0)


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=False,
    hyperbolic_bias=False,
    nonlin=None,
    c=1.0,
):
    """
    Mobius Linear operation for the Poincare ball model
    """

    if hyperbolic_input == True:
        output = ball.mobius_matvec(weight, input)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = ball.expmap0(output)
    if bias is not None:
        if hyperbolic_bias == False:
            bias = ball.expmap0(bias)
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.mobius_fn_apply(nonlin, output)
    output = ball.projx(output)
    return output


class MobLinear(torch.nn.Linear):
    """
    Mobius Linear class for the Poincare ball model, inherits 
    from nn.Linear module
    """

    def __init__(
        self,
        *args,
        hyperbolic_input=False,
        hyperbolic_bias=False,
        nonlin=None,
        c=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias == True:
                self.ball = manifold = geoopt.PoincareBall(c=c)
                self.bias = geoopt.ManifoldParameter(
                    self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(ball.expmap0(self.bias.normal_() / 4))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            c=1.0,
        )

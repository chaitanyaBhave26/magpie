/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "NeuralNetwork.h"
#include "MooseEnum.h"

registerMooseObject("MagpieApp", NeuralNetwork);

template <>
InputParameters
validParams<NeuralNetwork>()
{
  InputParameters params = NodalUserObject::validParams();
  params.addClassDescription("Reconstructs a neural network from a file and evaluates it");
  params.addRequiredParam<unsigned int>("H", "Number of neurons in each hidden layer");
  params.addRequiredParam<unsigned int>("N", "Number of hidden layers");
  params.addRequiredParam<unsigned int>("D_in", "Number of inputs to neural net");
  params.addRequiredParam<unsigned int>("D_out", "Number of outputs from neural net");
  params.addRequiredParam<FileName>("weights_file", "Name of the file with the neuron weights");
  MooseEnum activationFunctionEnum("SIGMOID SOFTSIGN TANH LINEAR", "SIGMOID");
  params.template addParam<MooseEnum>("activation_function",
                                      activationFunctionEnum,
                                      "Name of the hidden neuron activation function");
  params.addRequiredCoupledVar("variable", "Name of the variable this object operates on");
  return params;
}

NeuralNetwork::NeuralNetwork(const InputParameters & parameters)
  : NodalUserObject(parameters),
    _H(getParam<unsigned int>("H")),
    _N(getParam<unsigned int>("N")),
    _D_in(getParam<unsigned int>("D_in")),
    _D_out(getParam<unsigned int>("D_out")),
    _weights_file(getParam<FileName>("weights_file")),
    _activation_function(
        getParam<MooseEnum>("activation_function").template getEnum<ActivationFunction>()),
    MooseVariableInterface<Real>(this,
                                 false,
                                 "variable",
                                 Moose::VarKindType::VAR_ANY,
                                 Moose::VarFieldType::VAR_FIELD_STANDARD),
    _fe_vars(getCoupledMooseVars())
{

  // open the NN weights file
  setWeights();
  _inputs.resize(_D_in);
  _depend_vars.insert(name());
  for (unsigned int i = 0; i < _D_in; ++i)
  {
    auto temp = _fe_vars[i]->name();
    _depend_vars.insert(temp);
    _inputs[i] = &coupledValue("variable", i);
  }
}

void
NeuralNetwork::setWeights()
{
  std::ifstream ifile;
  std::string line;
  ifile.open(_weights_file);
  if (!ifile)
  {
    paramError("weights_file", "Unable to open file");
  }

  unsigned int line_no = 0;

  DenseMatrix<Real> _W_input(_H, _D_in);
  DenseVector<Real> _bias_input(_H);
  DenseMatrix<Real> _W_output(_D_out, _H);
  DenseVector<Real> _bias_output(_D_out);

  // Read input LINEAR neuron weights
  for (std::size_t i = 0; i < _H; i++)
  {
    for (std::size_t j = 0; j < _D_in; j++)
    {
      if (!(ifile >> _W_input(i, j)))
        mooseError("Error reading INPUT weights from file", _weights_file);
    }
  }
  for (std::size_t i = 0; i < _H; i++)
  {
    if (!(ifile >> _bias_input(i)))
      mooseError("Error reading  INPUT bias from file", _weights_file);
  }
  _weights.push_back(_W_input);
  _bias.push_back(_bias_input);

  for (int n = 0; n < _N -1; ++n) // For each hidden layer
    {
      DenseMatrix<Real> _W_hidden(_H, _H);
      DenseVector<Real> _bias_hidden(_H);

      for (std::size_t i = 0; i < _H; i++)
      {
        for (std::size_t j = 0; j < _H; j++)
        {
          if (!(ifile >> _W_hidden(i, j)))
            mooseError("Error reading HIDDEN weights from file", _weights_file);
        }
      }
      for (std::size_t i = 0; i < _H; i++)
      {
        if (!(ifile >> _bias_hidden(i)))
          mooseError("Error reading HIDDEN bias from file", _weights_file);
      }
      _weights.push_back(_W_hidden);
      _bias.push_back(_bias_hidden);
    }

  // Read OUTPUT LINEAR neuron weights
  for (unsigned int i = 0; i < _D_out; i++)
  {
    for (unsigned int j = 0; j < _H; j++)
    {
      if (!(ifile >> _W_output(i, j)))
        mooseError("Error reading OUTPUT weights from file", _weights_file);
    }
  }
  for (unsigned int i = 0; i < _D_out; i++)
  {
    if (!(ifile >> _bias_output(i) ))
      mooseError("Error reading OUTPUT bias from file", _weights_file);
  }
  _weights.push_back(_W_output);
  _bias.push_back(_bias_output);

}

void NeuralNetwork::finalize()
{
}

void NeuralNetwork::execute(/* arguments */) {}

void NeuralNetwork::initialize(/* arguments */) {}
void
NeuralNetwork::threadJoin(const UserObject & /*y*/)
{
}

const std::set<std::string> &
NeuralNetwork::getRequestedItems() const
{
  return _depend_vars;
}

Real
NeuralNetwork::eval(DenseVector<Real> & input, std::size_t op_id ) const
{
  // DenseVector<Real> input(_D_in);
  DenseVector<Real> feed_forward(_H);
  DenseVector<Real> temp(_H);
  DenseVector<Real> output(_D_out);

  // Apply input layer weights
  for (std::size_t i =0; i < _H; ++i)
  {
    feed_forward(i) = 0;
    for (std::size_t j = 0; j < _D_in; ++j)
      feed_forward(i)+=input(j)*_weights[0](i,j);
    feed_forward(i)+=_bias[0](i);

  }

  for (std::size_t n = 0; n < _N; ++n)
  {
    switch (_activation_function)
      {
        case ActivationFunction::SIGMOID:
          for (std::size_t i = 0; i < _H; ++i)
            feed_forward(i) = 1 / (1 + std::exp(-1 * feed_forward(i) ));

        case ActivationFunction::TANH:
          for (std::size_t i = 0; i < _H; ++i)
            feed_forward(i) = std::tanh(feed_forward(i) );
      }

      //bail out of linear layer if we are at last hidden layer
      if (n+1 == _N)
        break;
      //Applying connectivity linear layer
      for (std::size_t i =0; i < _H; ++i)
        {
        temp(i) = 0;
        for (std::size_t j = 0; j < _H; ++j)
          temp(i) += feed_forward(j)*_weights[n+1](i,j);
        temp(i)+= _bias[n+1](i);
        }
      feed_forward = temp;

  }

  //Apply final output layer
  auto n = _weights.size() - 1;
  for (std::size_t i =0; i < _D_out; ++i)
  {
    output(i) = 0;
    for (std::size_t j = 0; j < _H; ++j)
      output(i)+=feed_forward(j)*_weights[n](i,j);
    output(i)+=_bias[n](i);
  }
    if (output(op_id) < 0.8)
      {
        return 0.8;
      }
    else if (output(op_id) > 0.99)
      {
        return 0.99;
      }
    return output( op_id);
  }

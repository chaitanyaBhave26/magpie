//* This file is part of the MOOSE framework
//* https://www.mooseframework.org
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html

// #include "MooseMesh.h"
// #include "Conversion.h"
// #include "NonlinearSystem.h"
// #include <Eigen/Dense>
#include "NeuralNetwork.h"
#include "MooseEnum.h"
// #include <fstream>
// #include "libmesh/mesh_tools.h"
// #include "libmesh/point.h"
// #include "SystemBase.h"


registerMooseObject("MagpieApp", NeuralNetwork);

template<>
InputParameters
validParams<NeuralNetwork>()
{
  InputParameters params = NodalUserObject::validParams();
  params.addClassDescription("Reconstructs a neural network from a file and evaluates it");
  params.addRequiredParam<unsigned int>("H","Number of neurons in each hidden layer");
  params.addRequiredParam<unsigned int>("N","Number of hidden layers");
  params.addRequiredParam<unsigned int>("D_in","Number of inputs to neural net");
  params.addRequiredParam<unsigned int>("D_out","Number of outputs from neural net");
  params.addRequiredParam<FileName>("weights_file","Name of the file with the neuron weights");
  MooseEnum activationFunctionEnum("SIGMOID TANH","SIGMOID");
  params.template addParam<MooseEnum>("activation_function",activationFunctionEnum,"Name of the hidden neuron activation function");
  params.addRequiredParam<std::vector <NonlinearVariableName>>("variables","List of non-linear variables to be used as input");
  params.addRequiredCoupledVar("variable","Name of the variable this object operates on");
  params.addParam<std::vector<std::string>>("IC_dependencies","List of ICs that need to be executed before this UO");
  return params;
}

NeuralNetwork::NeuralNetwork(const InputParameters & parameters)
  : NodalUserObject(parameters),
    _H(getParam<unsigned int>("H")),
    _N(getParam<unsigned int>("N")),
    _D_in(getParam<unsigned int>("D_in")),
    _D_out(getParam<unsigned int>("D_out")),
    _weights_file(getParam<FileName>("weights_file")),
    _activation_function(getParam<MooseEnum>("activation_function").template getEnum<ActivationFunction>() ),
    _variables(getParam<std::vector <NonlinearVariableName>>("variables")  ),
    _ic_dependencies(getParam<std::vector<std::string>>("IC_dependencies")),
    MooseVariableInterface<Real>(this,
                               false,
                               "variables",
                               Moose::VarKindType::VAR_ANY,
                               Moose::VarFieldType::VAR_FIELD_STANDARD),
    _var(*mooseVariable()),
    _u(_var.dofValues()),
    // _var_vals(coupledValue("variable",1)),
    _fe_vars(getCoupledMooseVars())
{

  //open the NN weights file
  getWeights();
  // unsigned int n_variables = _variables.size();
  _inputs.resize(_D_in);
  _depend_vars.insert(name());
  for(unsigned int i =0; i < _fe_vars.size(); ++i)
    {
      auto temp = _fe_vars[i]->name();
      _depend_vars.insert(temp);
      _inputs[i] = &coupledValue("variable",i) ;
    }
  for(unsigned int i = 0; i < _ic_dependencies.size(); ++i)
    {
      _depend_vars.insert(_ic_dependencies[i]);
    }
  // _var(_sys.getActualFieldVariable(parameters.get<THREAD_ID>("_tid"), _variables[0]
                                        // );

// ApplyLinearInput();
  //convert

}
//
const std::set<std::string> &
NeuralNetwork::getRequestedItems()  const
{
  std::cout << "Someone asked for this";
  return _depend_vars;
}
void NeuralNetwork::getWeights()
  {
    std::ifstream ifile;
    std::string line;
    ifile.open(_weights_file);
    if(!ifile)
      {
        paramError("weights_file","Unable to open file");
      }

    unsigned int line_no = 0;

    // _weights.resize(_N);

    std::vector<std::vector <Real>> _W_input(_H,std::vector <Real> (_D_in,0));
    std::vector<Real> _bias_input(_H,0);
    std::vector<std::vector <Real>> _W_output(_D_out,std::vector <Real> (_H,0));
    std::vector<Real> _bias_output(_D_out);



    while(std::getline(ifile,line) )
      {
          Real val = std::stod(line);
          if( line_no < _H*_D_in)
            {
              int j = line_no%_D_in;
              int i = (line_no - j)/_D_in;

              _W_input[i][j] = val;
            }
          else if( line_no < _H*_D_in + _H)
            {
              int i = line_no - _H*_D_in;
              _bias_input[i] = val;
            }
            else if( line_no < _H*_D_in + _H + _H*_D_out)
              {
                int temp = line_no -_H*_D_in - _H;
                int j = temp%_D_out;
                int i = (temp - j)/_D_out;
                _W_output[j][i] = val;
                // std::cout << _W_output[j][i] << "\n";
              }
            else
            {
               int temp = line_no < _H*_D_in + _H + _H*_D_out;
               _bias_output[temp] = val;
            }
          // std::istringstream iss(line);


          // std::cout << line;
          line_no ++;
      }

    _weights.push_back(_W_input);
    _weights.push_back(_W_output);
    _bias.push_back(_bias_input);
    _bias.push_back(_bias_output);
  }


void
NeuralNetwork::finalize(/* arguments */) {
  /* code */
}

void
NeuralNetwork::execute(/* arguments */) {
  /* code */
  // retval = eval
  // unsigned int nodeID = _current_node->id();
  // unsigned int ii = map_MOOSE2Ext(*_current_node);
  // _ext_data[ii] = _u[0];
  // auto _var_test = ;
  // std::cout<< _fe_vars[0]->name() << ", ";
}

void
NeuralNetwork::initialize(/* arguments */) {
  /* code */
}
void
NeuralNetwork::threadJoin(const UserObject & /*y*/)
{
  // I don't know what to do with this yet.
}

Real
NeuralNetwork::eval( ) const
{

  // NN_eval();
  std::vector<Real> ip_vect;
  std::vector<Real> lin_output;
  for (int i =0; i<_D_in; ++i)
    {
      auto *temp = _inputs[i];
      ip_vect.push_back(temp[0][0]);
    }
  ApplyLinearInput(ip_vect,lin_output);

  //Apply sigmoids
  for (int j=0; j < _N; ++j)
  {
    // std::cout << "Layer" << j << ":\n";
    switch (_activation_function)
    {
      case ActivationFunction::SIGMOID:
        for (int i =0; i < _H; ++i)
          {
            Real temp = 1/(1 + std::exp(-1*lin_output[i]) );
            lin_output[i] = temp;
            // std::cout << temp << "\t";
          }
          // std::cout << "\n";
    }
  }

  Real final_output;
  ApplyLinearOutput(lin_output,final_output);


  // return  *temp[0];
  return final_output;
}

void
NeuralNetwork::ApplyLinearInput( std::vector<Real> & input,std::vector<Real> & output) const
  {
    // std::cout << "Inputs " << input.size() << "\n";
    output.resize(_H);
    // Output = Input*Weights
    for (int i=0; i< _H; ++i)
      {
        Real temp = 0;
        for (int j = 0; j< _D_in; ++j)
          {
            temp+=input[j]*_weights[0][i][j];
          }
        output[i] = temp + _bias[0][i];

      }

  }

void
NeuralNetwork::ApplyLinearOutput( std::vector<Real> & input,Real & output) const
  {
    // std::cout << "Inputs " << input.size() << "\n";
    if (! (_D_out == 1) )
      {
        mooseError("No implementation for an NN with more than 1 outputs yet");
      }
    // Output = Input*Weights
    for (int i=0; i< _H; ++i)
      {
        output+=input[i]*_weights[1][0][i];

      }
      output += _bias[1][0];
      if (output >= 1.0)
        {
          output = 1.0 - 1e-3;
        }
      else if (output <= 0.0)
        {
          output = 1e-3;
        }
  }

// void NeuralNetwork::NN_eval()
// {
//
//   // std::cout << temp[0][0] << "\n";
//   // return 0.0;
//
// }

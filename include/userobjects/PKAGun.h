/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/* MAGPIE - Mesoscale Atomistic Glue Program for Integrated Execution */
/*                                                                    */
/*            Copyright 2017 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "PKAFixedPointGenerator.h"

class PKAGun;

template <>
InputParameters validParams<PKAGun>();

/**
 * Starts PKAs at a fixed point in a fixed direction
 */
class PKAGun : public PKAFixedPointGenerator
{
public:
  PKAGun(const InputParameters & parameters);

protected:
  /// provides a mean to override the angular distribution of the PKAs in derived class
  virtual void setDirection(MyTRIM_NS::IonBase & ion) const;

  /// the direction along which the PKAs move
  const Point _direction;
};


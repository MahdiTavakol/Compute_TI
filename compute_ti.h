/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(thermo_integ,ComputeTI);
// clang-format on
#else

#ifndef COMPUTE_THERMO_INTEG_H
#define COMPUTE_THERMO_INTEG_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeThermoInteg : public Compute {
 public:
  ComputeThermoInteg(class LAMMPS *, int, char **);
  ~ComputeThermoInteg() override;
  void init() override;
  void compute_vector() override;

 private:
  // Parameters
  int parameter_list;

  double delta;
  double typeA, typeB, typeC;

  // Pair style parameters
  char * pstyle, * pparam;
  Pair * pair;
  int pdim1;

  struct selected_types {int typeA;
	                 int typeB; 
	                 int typeC;
                         int countA;
                         int countB;
                         int countC;};
  selected_types selected;

  template <int parameter, int mode>  
  double compute_du(int &delta);
  void allocate_storage();
  void deallocate_storage();
  template  <int direction>
  void forward_reverse_copy(double &,double &);
  template  <int direction>
  void forward_reverse_copy(double* ,double* , int );
  template  <int direction>
  void forward_reverse_copy(double** ,double** , int , int );
  template <int direction>
  void backup_restore_qfev();
  template <int parameter, int mode>   
  void modify_epsilon_q(double& delta);
  void count_atoms(selected_type selected);
  void update_lmp();
  void compute_q_total();
  void compute_epair();
};

}    // namespace LAMMPS_NS

#endif
#endif

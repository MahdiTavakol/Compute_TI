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
ComputeStyle(thermo_integ,ComputeThermoInteg);
// clang-format on
#else

#ifndef COMPUTE_THERMO_INTEG_H
#define COMPUTE_THERMO_INTEG_H

#include "compute.h"
#include "pair.h"

namespace LAMMPS_NS {

class ComputeThermoInteg : public Compute {
 public:
  ComputeThermoInteg(class LAMMPS *, int, char **);
  ~ComputeThermoInteg() override;
  void init() override;
  void setup() override;
  void compute_vector() override;

 private:
  int mode;
  // Parameters
  int parameter_list;

  double delta_p, delta_q;
  double typeA, typeB, typeC;

  // Pair style parameters
  char * pstyle, * pparam;
  Pair * pair;
  int pdim;

  struct selected_types {int typeA;
	                 int typeB; 
	                 int typeC;
                         int countA;
                         int countB;
                         int countC;};
  selected_types selected;

  class Fix *fixgpu;

  // This is just a pointer to the non-bonded interaction parameters and does not have any allocated memory
  // This should not be deallocated since the original pointer will be deallocated later on by the LAMMPS
  double **epsilon;
  // _init is the initial value of hydrogen atoms properties which is multiplied by lambda at each step
  double **epsilon_init;

  // _org is for value of parameters before the update_lmp() with modified parameters act on them
  double *q_orig;
  double **f_orig;
  double eng_vdwl_orig, eng_coul_orig;
  double pvirial_orig[6];
  double *peatom_orig, **pvatom_orig;
  double energy_orig;
  double kvirial_orig[6];
  double *keatom_orig, **kvatom_orig;

  int nmax;


  template <int parameter, int mode>  
  double compute_du(double &delta);
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
  void count_atoms(selected_types selected);
  void update_lmp();
  void compute_q_total();
  double compute_epair();
};

}    // namespace LAMMPS_NS

#endif
#endif

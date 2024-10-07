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
  void compute_peratom() override {}; // I just wanted LAMMPS to consider this as peratom compute so the peratom energies be tallied in this timestep.
  int pack_reverse_comm(int , int , double *) override;
  void unpack_reverse_comm(int , int *, double *) override;
  
  
  
 private:
  int mode;
  // Parameters
  int parameter_list;

  double lambda, dlambda;

  int ntypeAs, ntypeBs;
  int * typeAs, * typeBs;
  double * p_initials, * p_finals;
  double * q_initials, * q_finals;
  double * delta_ps, * delta_qs;
  double * lA_ps, * lB_ps;
  double * lA_qs, * lB_qs;

  double  delta_qC;
  int typeC;

  // Pair style parameters
  char * pstyle, * pparam;
  Pair * pair;
  int pdim;


  class Fix *fixgpu;

  // This is just a pointer to the non-bonded interaction parameters and does not have any allocated memory
  // This should not be deallocated since the original pointer will be deallocated later on by the LAMMPS
  double **epsilon;
  // _init is the initial value of selected atom properties which is multiplied by lambda at each step
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
  bigint natoms;

  // Energy of each atom computed for states A and B
  double *energy_peratom;



  void allocate_storage();
  void deallocate_storage();

  template  <int direction>
  static void forward_reverse_copy(double &,double &);
  template  <int direction>
  static void forward_reverse_copy(double* ,double* , int );
  template  <int direction>
  static void forward_reverse_copy(double** ,double** , int , int );
  template <int direction>
  void backup_restore_qfev();


  template <int parameter, int mode>  
  double compute_du(double &delta_p, double &delta_q);
  template <int parameter, int mode>   
  void modify_epsilon_q(double& delta_p, double& delta_q);
  void restore_epsilon();
  void update_lmp();
  void set_delta_qC(const double & _delta_q, double & _delta_qC);
  void compute_q_total();
  double compute_epair();
  double compute_epair_atom();
};

}    // namespace LAMMPS_NS

#endif
#endif

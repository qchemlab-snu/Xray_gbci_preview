import numpy
from pyscf.sfnoci import sfnoci
from pyscf import fci
from pyscf.data import nist
from pyscf import lib
from fcisiso import SU2CG
from fcisiso import get_jk, get_jk_amfi, compute_hso_ao
from xray_gbci.gbci_siso import make_trans, make_trans_rdm1
from mpi4py import MPI

def kernel_siso_we(sfnoci, ci = None, po_list = None, group = None, ov_list = None, hsoao =None, dmao=None, amfi=True):
    mol = sfnoci.mol
    ncas = sfnoci.ncas
    ncore = sfnoci.ncore
    if ci is None: 
        raise NotImplementedError
    if po_list is None:
        raise NotImplementedError
    if ov_list is None:
        raise NotImplementedError   
    if hsoao is None:
        print('\nGenerating Spin-Orbit Integrals:\n')
        gsci = 0
        for index, ici in enumerate(ci):
            if ici[4] < ici[gsci][4]:
                    gsci = index
        if dmao is None: 
            dmao = sfnoci.make_rdm1(ci[gsci][5], sfnoci.mo_coeff, nelecas= (ci[gsci][0],ci[gsci][1]))
        hsoao = compute_hso_ao(mol, dmao, amfi = amfi) *2
    hso = numpy.einsum('rij,ip,jq -> rpq',hsoao, sfnoci.mo_coeff[:,ncore:ncore + ncas],sfnoci.mo_coeff[:,ncore:ncore + ncas])
    hso_pmz = numpy.zeros_like(hso)
    hso_pmz[0] = (1j*hso[1] - hso[0])/2
    hso_pmz[1] = (1j*hso[1] + hso[0])/2
    hso_pmz[2] = hso[2]*numpy.sqrt(0.5)
    
    # state interaction
    #
    su2cg = SU2CG()
    #Spin multiplicity list
    ms_dim = [ici[2]+1 for ici in ci]
    #index shift
    idx_shift = [sum(ms_dim[:i]) for i in range(len(ms_dim))]
    hdiag = numpy.array([ici[4] for ici in ci for ms in range(ici[2]+1)], dtype=complex)
    hsiso = numpy.zeros((hdiag.shape[0], hdiag.shape[0]), dtype=complex)
    thrds = 29.0  # cm-1
    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
    print("\nComplex SO-Hamiltonian matrix elements over spin components of spin-free eigenstates:")
    print("(In cm-1. Print threshold:%10.3f cm-1)\n" % thrds)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
   
    local_hsiso = numpy.zeros_like(hsiso)
    def divide_into_nodes(ci, N):
        chunk_size = len(ci) // N
        remainder = len(ci) % N  

        indices = []
        start = 0

        for i in range(N):
            end = start + chunk_size + (1 if i < remainder else 0)
            indices.append(range(start, end))
            start = end
        return indices
    istate_list = divide_into_nodes(ci, size)
    istates = istate_list[rank]
    for istate in istates:
        ici = ci[istate]
        for jstate, jci in enumerate(ci):
            if jstate < istate:
                continue
            zero_me_ij = False
            if abs(ici[2]-jci[2]) > 2:
                zero_me_ij = True
            elif ici[2] == jci[2] == 0:
                zero_me_ij = True
            elif abs(ici[3]-jci[3]) > 2:
                zero_me_ij = True
            elif jci[3] == ici[3] - 2:
                CGcoeff = su2cg.clebsch_gordan(ici[2], 2, jci[2], ici[3], -2, jci[3])
                if CGcoeff == 0:
                    zero_me_ij = True
                else:
                    tp1 = make_trans(1, ici[-1], jci[-1],
                            ncas, ici[:2], jci[:2],po_list, group, ov_list)
                    ij_red_den = tp1 / CGcoeff
            elif jci[3] == ici[3] + 2:
                CGcoeff = su2cg.clebsch_gordan(ici[2], 2, jci[2], ici[3], 2, jci[3])
                if CGcoeff == 0:
                    zero_me_ij = True
                else:
                    tm1 = make_trans(-1, ici[-1], jci[-1],
                            ncas, ici[:2], jci[:2], po_list, group, ov_list)
                    ij_red_den = tm1 / CGcoeff
            elif jci[3] == ici[3]:
                CGcoeff = su2cg.clebsch_gordan(ici[2], 2, jci[2], ici[3], 0, jci[3])
                if CGcoeff == 0:
                    zero_me_ij = True
                else:
                    tze = make_trans(0, ici[-1], jci[-1],
                            ncas, ici[:2], jci[:2],po_list, group, ov_list)
                    ij_red_den = tze / CGcoeff
                    print(ij_red_den)
            else:
                zero_me_ij = True
            for ii, i_ms2 in enumerate(range(-ici[2], ici[2] + 1, 2)):
                for jj, j_ms2 in enumerate(range(-jci[2], jci[2] + 1, 2)):
                    if zero_me_ij or abs(i_ms2 - j_ms2) > 2 :
                        somat = 0
                    elif j_ms2 == i_ms2 - 2:
                        CGcoeff = su2cg.clebsch_gordan(ici[2], 2, jci[2], i_ms2, -2, j_ms2)
                        somat = numpy.einsum('ij,ij->', ij_red_den, hso_pmz[0])*CGcoeff
                    elif j_ms2 == i_ms2 + 2:
                        CGcoeff = su2cg.clebsch_gordan(ici[2], 2, jci[2], i_ms2, 2, j_ms2)
                        somat = numpy.einsum('ij,ij->', ij_red_den, hso_pmz[1])*CGcoeff
                    elif j_ms2 == i_ms2:
                        CGcoeff = su2cg.clebsch_gordan(ici[2], 2, jci[2], i_ms2, 0, j_ms2)
                        somat = numpy.einsum('ij,ij->', ij_red_den, hso_pmz[2])*CGcoeff
                    else:
                        somat = 0
                    local_hsiso[idx_shift[istate]+ii, idx_shift[jstate]+jj] = somat
                    if istate != jstate:
                        local_hsiso[idx_shift[jstate]+jj, idx_shift[istate]+ii] = numpy.conj(somat)
                    somat *= au2cm
                    if abs(somat) > thrds:
                        print(('<%4d|H_SO|%4d> I1 = %4d (E1 = %15.8f) S1 = %4.1f MS1 = %4.1f '
                                   + 'I2 = %4d (E2 = %15.8f) S2 = %4.1f MS2 = %4.1f '
                                   + 'Re = %9.3f Im = %9.3f')
                                  % (idx_shift[istate]+ii, idx_shift[jstate]+jj,
                                     istate, ici[4], ici[2] / 2, i_ms2 / 2,
                                     jstate, jci[4], jci[2] / 2, j_ms2 / 2,
                                     somat.real, somat.imag))
            comm.Reduce(local_hsiso, hsiso, op = MPI.SUM, root = 0)

    if rank == 0: 
        # Full SISO Hamiltonian eigen states
        hfull = hsiso + numpy.diag(hdiag)
        heig, hvec = numpy.linalg.eigh(hfull)
        print('\nTotal energies including SO-coupling:\n')
        for i in range(len(heig)):
            sf_proj_vec = []
            for ibra, mult in enumerate(ms_dim):
                sf_proj_vec.append(numpy.linalg.norm(
                    hvec[idx_shift[ibra]:idx_shift[ibra] + mult, i]) ** 2)
            iv = numpy.argmax(numpy.abs(sf_proj_vec))
            print(('  State %4d Total energy: %15.8f | largest |proj_norm|**2 % 6.4f '
                + 'from I = %4d E = %15.8f S = %4.1f')
                % (i, heig[i], sf_proj_vec[iv], iv, ci[iv][4], ci[iv][2] / 2))
            return heig, hvec
        else:
            return None, None
        

def dipole_sdm_coreex_we(sfnoci, ci_ex, ci_gr, po_list, group, ov_list):
        mol = sfnoci.mol
        ncas = sfnoci.ncas
        ncore = sfnoci.ncore
        su2cg = SU2CG()
        ms_dim_ex = [ci[2]+1 for ci in ci_ex]
        idx_shift_ex = [sum(ms_dim_ex[:i]) for i in range(len(ms_dim_ex))]
        ms_dim_gr = [ci[2]+1 for ci in ci_gr]
        idx_shift_gr = [sum(ms_dim_gr[:i]) for i in range(len(ms_dim_gr))]
        dip_ints = mol.intor_symmetric('int1e_r')
        dipole_sdm = numpy.zeros((3,sum(ms_dim_ex),sum(ms_dim_gr)))
        ##dipole calculation###

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_dipole_sdm = numpy.zeros_like(dipole_sdm)
        def divide_into_nodes(ci, N):
            chunk_size = len(ci) // N
            remainder = len(ci) % N 
            indices = []
            start = 0

            for i in range(N):
                end = start + chunk_size + (1 if i < remainder else 0)
                indices.append(range(start, end))
                start = end
            return indices
        istate_list = divide_into_nodes(ci_ex, size)
        istates = istate_list[rank]
        for istate in istates:
            ici = ci_ex[istate]
            for jstate, jci in enumerate(ci_gr):
                if ici[2] == jci[2]:
                    if ici[3] != jci[3]: NotImplementedError
                    elif ici[3] == jci[3]:
                        zero_me_ij = False   
                        CGcoeff = su2cg.clebsch_gordan(ici[2], 0, jci[2], ici[3], 0, jci[3])
                        if CGcoeff == 0:
                            zero_me_ij = True
                        else:
                            rdm1a = make_trans_rdm1('aa', ici[-1], jci[-1], ncas, (ici[0], ici[1]), (jci[0],jci[1]),po_list, group, ov_list)
                            rdm1b = make_trans_rdm1('bb', ici[-1], jci[-1], ncas, (ici[0], ici[1]), (jci[0],jci[1]),po_list, group, ov_list)
                            trans_rdm = rdm1a + rdm1b
                            mo_cas = sfnoci.mo_coeff[:,ncore:ncore + ncas]
                            trans_rdm = lib.einsum('ij, ai, bj -> ab', trans_rdm, mo_cas, mo_cas)
                            ij_red_den = trans_rdm / CGcoeff
                    for ii, i_ms2 in enumerate(range(-ici[2], ici[2] + 1, 2)):
                        for jj, j_ms2 in enumerate(range(-jci[2], jci[2] + 1, 2)):
                            if not zero_me_ij and i_ms2 == j_ms2:
                               CGcoeff = su2cg.clebsch_gordan(ici[2], 0, jci[2], i_ms2, 0, j_ms2)
                               local_dipole_sdm[:,idx_shift_ex[istate]+ii,idx_shift_gr[jstate]+jj] += lib.einsum('xij,ij ->x',dip_ints, ij_red_den) * CGcoeff
        comm.Reduce(local_dipole_sdm, dipole_sdm, op = MPI.SUM, root = 0)
        if rank == 0:
            return dipole_sdm
        else:
            return None

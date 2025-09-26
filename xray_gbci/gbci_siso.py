import numpy
from pyscf.sfnoci import sfnoci
from pyscf import fci
from pyscf.data import nist
from pyscf import lib
from fcisiso import SU2CG
from fcisiso import get_jk, get_jk_amfi, compute_hso_ao

def make_trans(m, cibra, ciket, norb, nelec_bra, nelec_ket,po_list,group,ov_list):
    if m == 1:
        return -1.0 * make_trans_rdm1('ab', cibra, ciket, norb, nelec_bra, nelec_ket, po_list, group, ov_list)
    elif m == -1:
        return make_trans_rdm1('ba', cibra, ciket, norb, nelec_bra, nelec_ket, po_list, group, ov_list)
    else:
        return numpy.sqrt(0.5) * (make_trans_rdm1('aa', cibra, ciket, norb, nelec_bra, nelec_ket, po_list, group, ov_list)
                               - make_trans_rdm1('bb', cibra, ciket, norb, nelec_bra, nelec_ket, po_list, group,ov_list))
    
def make_trans_rdm1(dspin, cibra, ciket, norb, nelec_bra,nelec_ket, po_list, group, ov_list):
    """
    One-particle transition density matrix between states
    with different spins.

    Args:
        dspin : str 'aa' or 'bb' or 'ab' or 'ba'
            the spin subscript of p, q operators
        cibra : np.ndarray((n_det_alpha, n_det_beta))
            ci vector for bra wavefunction
        ciket : np.ndarray((n_det_alpha, n_det_beta))
            ci vector for ket wavefunction
        norb : int
            number of orbitals
            in this case, number of active orbitals
        nelec_bra : (int, int)
            numebr of alpha and beta electrons in bra state
        nelec_ket : (int, int)
            numebr of alpha and beta electrons in ket state
        PO : (int, int)
            possible occupation number pattern of active space
        group: list or None
            list : SF-NOCI with grouped bath
            None : SF-NOCI        
    Returns:
        rdm1 : np.ndarray((norb, norb))
            transition density matrix
    """
    nelabra, nelbbra = nelec_bra
    nelaket, nelbket = nelec_ket
    if dspin == 'ba':
        cond = nelabra == nelaket - 1 and nelbbra == nelbket + 1
    elif dspin == 'ab':
        cond = nelabra == nelaket + 1 and nelbbra == nelbket - 1
    elif dspin == 'aa':
        cond = nelabra == nelaket and nelbbra == nelbket and nelabra > 0
    else:
        cond = nelabra == nelaket and nelbbra == nelbket and nelbbra > 0
    if not cond:
        return numpy.array(0)
    nabra = fci.cistring.num_strings(norb, nelabra)
    nbbra = fci.cistring.num_strings(norb, nelbbra)
    naket = fci.cistring.num_strings(norb, nelaket)
    nbket = fci.cistring.num_strings(norb, nelbket)
    ketstringa = fci.cistring.make_strings(range(norb), nelaket)
    ketstringb = fci.cistring.make_strings(range(norb), nelbket)
    brastringa = fci.cistring.make_strings(range(norb), nelabra)
    brastringb = fci.cistring.make_strings(range(norb), nelbbra)
    cibra = cibra.reshape(nabra, nbbra)
    ciket = ciket.reshape(naket, nbket)
    rdm1 = numpy.zeros((norb,norb))
    if cond:
        if dspin == 'ba':
            lidxa = fci.cistring.gen_des_str_index(range(norb),nelaket)
            lidxb = fci.cistring.gen_cre_str_index(range(norb),nelbket)
            for str0a in range(naket):
                for str0b in range(nbket):
                    for a, _, str1b, signb in lidxb[str0b]:
                        for _ , i, str1a, signa in lidxa[str0a]:
                            w_occa = sfnoci.str2occ(ketstringa[str0a],norb)
                            w_occb = sfnoci.str2occ(ketstringb[str0b],norb)
                            x_occa = sfnoci.str2occ(brastringa[str1a],norb)
                            x_occb = sfnoci.str2occ(brastringb[str1b],norb)
                            x_occ = x_occa + x_occb
                            w_occ = w_occa + w_occb
                            p1 = sfnoci.find_matching_rows(po_list,x_occ)
                            p2 = sfnoci.find_matching_rows(po_list,w_occ)
                            if group is not None:
                                p1 = sfnoci.num_to_group(group,p1)
                                p2 = sfnoci.num_to_group(group,p2)
                            rdm1[a,i] += numpy.conjugate(cibra[str1a,str1b])*ciket[str0a,str0b]*ov_list[p1,p2]*signa *signb

        elif dspin == 'ab':
            lidxa = fci.cistring.gen_cre_str_index(range(norb),nelaket)
            lidxb = fci.cistring.gen_des_str_index(range(norb),nelbket)
            for str0a in range(naket):
                for str0b in range(nbket):
                    for a, _, str1a, signa in lidxa[str0a]:
                        for _, i, str1b, signb in lidxb[str0b]:
                            w_occa = sfnoci.str2occ(ketstringa[str0a],norb)
                            w_occb = sfnoci.str2occ(ketstringb[str0b],norb)
                            x_occa = sfnoci.str2occ(brastringa[str1a],norb)
                            x_occb = sfnoci.str2occ(brastringb[str1b],norb)
                            x_occ = x_occa + x_occb
                            w_occ = w_occa + w_occb
                            p1 = sfnoci.find_matching_rows(po_list,x_occ)
                            p2 = sfnoci.find_matching_rows(po_list,w_occ)
                            if group is not None:
                                p1 = sfnoci.num_to_group(group,p1)
                                p2 = sfnoci.num_to_group(group,p2)
                            rdm1[a,i] += numpy.conjugate(cibra[str1a,str1b])*ciket[str0a,str0b]*ov_list[p1,p2]*signa *signb

        elif dspin == 'aa':
            lidxa = fci.cistring.gen_linkstr_index(range(norb),nelaket)
            for str0a in range(naket):
                for a, i, str1a, signa in lidxa[str0a]:
                    for str0b in range(nbket):
                        w_occa = sfnoci.str2occ(ketstringa[str0a],norb)
                        w_occb = sfnoci.str2occ(ketstringb[str0b],norb)
                        x_occa = sfnoci.str2occ(brastringa[str1a],norb)
                        x_occ = x_occa + w_occb
                        w_occ = w_occa + w_occb
                        p1 = sfnoci.find_matching_rows(po_list,x_occ)
                        p2 = sfnoci.find_matching_rows(po_list,w_occ)
                        if group is not None:
                            p1 = sfnoci.num_to_group(group,p1)
                            p2 = sfnoci.num_to_group(group,p2)
                        rdm1[a,i] += numpy.conjugate(cibra[str1a,str0b])*ciket[str0a,str0b]*ov_list[p1,p2]*signa
        
        elif dspin == 'bb':
            lidxb = fci.cistring.gen_linkstr_index(range(norb),nelbket)
            for str0b in range(nbket):
                for a, i , str1b, signb in lidxb[str0b]:
                    for str0a in range(naket):
                        w_occa = sfnoci.str2occ(ketstringa[str0a],norb)
                        w_occb = sfnoci.str2occ(ketstringb[str0b],norb)
                        x_occb = sfnoci.str2occ(brastringb[str1b],norb)
                        x_occ = w_occa + x_occb
                        w_occ = w_occa + w_occb
                        p1 = sfnoci.find_matching_rows(po_list,x_occ)
                        p2 = sfnoci.find_matching_rows(po_list,w_occ)
                        if group is not None:
                            p1 = sfnoci.num_to_group(group,p1)
                            p2 = sfnoci.num_to_group(group,p2)
                        rdm1[a,i] += numpy.conjugate(cibra[str0a,str1b])*ciket[str0a,str0b]* ov_list[p1,p2]*signb
        return rdm1
    
def kernel_siso_we(sfnoci, ci = None, po_list = None, group = None, ov_list = None, hsoao = None, amfi = True):
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

    for istate, ici in enumerate(ci):
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
                    hsiso[idx_shift[istate]+ii, idx_shift[jstate]+jj] = somat
                    if istate != jstate:
                        hsiso[idx_shift[jstate]+jj, idx_shift[istate]+ii] = numpy.conj(somat)
                    somat *= au2cm
                    if abs(somat) > thrds:
                        print(('<%4d|H_SO|%4d> I1 = %4d (E1 = %15.8f) S1 = %4.1f MS1 = %4.1f '
                                   + 'I2 = %4d (E2 = %15.8f) S2 = %4.1f MS2 = %4.1f '
                                   + 'Re = %9.3f Im = %9.3f')
                                  % (idx_shift[istate]+ii, idx_shift[jstate]+jj,
                                     istate, ici[4], ici[2] / 2, i_ms2 / 2,
                                     jstate, jci[4], jci[2] / 2, j_ms2 / 2,
                                     somat.real, somat.imag))
    hfull = hsiso + numpy.diag(hdiag)
    heig, hvec = numpy.linalg.eigh(hfull)
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

def dipole_sdm_coreex_we(sfnoci,ci_ex,ci_gr,po_list, group, ov_list):
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
    for istate, ici in enumerate(ci_ex):
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
                            dipole_sdm[:,idx_shift_ex[istate]+ii,idx_shift_gr[jstate]+jj] += lib.einsum('xij,ij ->x',dip_ints, ij_red_den) * CGcoeff
    return dipole_sdm




import numpy
from pyscf import fci
from pyscf.fci import cistring
from pyscf.fci import fci_slow
from pyscf.fci.direct_spin1 import spin_op
from pyscf.fci import selected_ci
from pyscf import lib
from pyscf.sfnoci.sfnoci import str2occ, find_matching_rows, num_to_group

def construct_hamiltonian_matrix(nelec,ncas,h1eff,eri,po_list,group, ov_list,ecore_list):
    stringsa = cistring.make_strings(range(ncas),nelec[0])
    stringsb = cistring.make_strings(range(ncas),nelec[1])
    link_indexa = cistring.gen_linkstr_index(range(ncas),nelec[0])
    link_indexb = cistring.gen_linkstr_index(range(ncas),nelec[1])
    na = cistring.num_strings(ncas,nelec[0])
    nb = cistring.num_strings(ncas,nelec[1])
    idx_a = numpy.arange(na)
    idx_b = numpy.arange(nb)
    mat1 = numpy.zeros((na,nb,na,nb))
    matTSc = numpy.zeros((na,nb,na,nb))
    for str0a, taba in enumerate(link_indexa):
        for pa, qa, str1a, signa in taba:
            for str0b, tabb in enumerate(link_indexb):
                for pb, qb, str1b, signb in tabb:
                    w_occa = str2occ(stringsa[str0a],ncas)
                    w_occb = str2occ(stringsb[str0b],ncas)
                    x_occa = str2occ(stringsa[str1a],ncas)
                    x_occb = str2occ(stringsb[str1b],ncas)
                    x_state_occ = numpy.array(x_occa) + numpy.array(x_occb)
                    w_state_occ = numpy.array(w_occa) + numpy.array(w_occb)
                    p1=find_matching_rows(po_list,x_state_occ)[0]
                    p2=find_matching_rows(po_list,w_state_occ)[0]
                    if group is not None: 
                       p1 = num_to_group(group,p1)
                       p2 = num_to_group(group,p2) 
                    if matTSc[str1a,str1b,str0a,str0b]==0:
                        matTSc[str1a,str1b,str0a,str0b] += ov_list[p1,p2]    
                    if pa==qa and pb ==qb:
                        mat1[str1a,str1b,str0a,str0b] += (signa * h1eff[p1,p2,pa,qa]/nelec[1]  + signb * h1eff[p1,p2,pb,qb]/nelec[0])
                    elif pa!=qa and pb == qb:
                        mat1[str1a,str1b,str0a,str0b] += signa * h1eff[p1,p2,pa,qa]/nelec[1]
                    elif pa==qa and pb !=qb:
                        mat1[str1a,str1b,str0a,str0b] += signb * h1eff[p1,p2,pb,qb]/nelec[0]
                    elif pa!=qa and pb !=qb:
                        mat1[str1a,str1b,str0a,str0b] += 0
                    #mat1[str1a,idx_b,str0a,idx_b] += signa * h1c[g1,g2,pa,qa]
                    #mat1[idx_a,str1b,idx_a,str0b] += signb * h1c[g1,g2,pb,qb]

    #mat1 = mat1.reshape(na*nb,na*nb)
    h2 = fci_slow.absorb_h1e(h1eff[0,0]*0, eri, ncas, nelec)
    t1 = numpy.zeros((ncas,ncas,na,nb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            # alpha spin
            t1[a,i,str1,idx_b,str0,idx_b] += sign
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            # beta spin
            t1[a,i,idx_a,str1,idx_a,str0] += sign
    t1 = lib.einsum('psqr,qrABab->psABab', h2, t1)
    mat2 = numpy.zeros((na,nb,na,nb))
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            # alpha spin
            mat2[str1] += sign * t1[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
           # beta spin
            mat2[:,str1] += sign * t1[a,i,:,str0]
    #mat2 = mat2.reshape(na*nb,na*nb)
    ham = (mat1+0.5*mat2)*matTSc
    ham = ham.reshape(na*nb,na*nb)
    K = numpy.zeros((na,nb))
    for i in range(0,na):
        for j in range(0,nb):
            x_occa = str2occ(stringsa[i],ncas)
            x_occb = str2occ(stringsb[j],ncas)        
            x_state_occ = numpy.array(x_occa) + numpy.array(x_occb)
            p1=find_matching_rows(po_list,x_state_occ)[0]
            if group is not None: 
                p1 = num_to_group(group,p1) 
            K[i,j] = ecore_list[p1]
    K = K.reshape(-1)
    K = numpy.diag(K)
    # print("mat1")
    # print(mat1.reshape(na*nb,na*nb)) 
    # print("mat2")
    # print(0.5*mat2.reshape(na*nb,na*nb))
    # print("K")
    # print(K)
    hamiltonian = ham + K 
    return hamiltonian

def make_non_excited_string(nras1, nras2, neleras2):
    ras1 = range(0, nras1)
    ras2 = range(nras1, nras1 + nras2)
    ras1string0 = cistring.make_strings(ras1, nras1)
    ras2string0 = cistring.make_strings(ras2, neleras2)
    rasstring0 = numpy.add.outer(ras1string0,ras2string0).flatten()
    return numpy.array(rasstring0, order = 'C', dtype = numpy.int64)

# def contract_spin_up(fcivec, norb, nelec, s2, ms2):
#     neleca, nelecb = fci.addons._unpack_nelec(nelec)
#     na = cistring.num_strings(norb, neleca)
#     nb = cistring.num_strings(norb, nelecb)
#     fcivec = fcivec.reshape(na,nb)

#     def gen_map(fstr_index, nelec, des=True):
#         a_index = fstr_index(range(norb), nelec)
#         amap = numpy.zeros((a_index.shape[0],norb,2), dtype=numpy.int32)
#         if des:
#             for k, tab in enumerate(a_index):
#                 amap[k,tab[:,1]] = tab[:,2:]
#         else:
#             for k, tab in enumerate(a_index):
#                 amap[k,tab[:,0]] = tab[:,2:]
#         return amap

    
#     if neleca > 0:
#         ades = gen_map(cistring.gen_des_str_index, neleca)
#     else:
#         ades = None

#     if nelecb > 0:
#         bdes = gen_map(cistring.gen_des_str_index, nelecb)
#     else:
#         bdes = None

#     if neleca < norb:
#         acre = gen_map(cistring.gen_cre_str_index, neleca, False)
#     else:
#         acre = None

#     if nelecb < norb:
#         bcre = gen_map(cistring.gen_cre_str_index, nelecb, False)
#     else:
#         bcre = None
#     def trans(ci_coeff, aindex, bindex, nea, neb):
#         if aindex is None or bindex is None:
#             return None

#         t1 = numpy.zeros((cistring.num_strings(norb,nea),
#                           cistring.num_strings(norb,neb)))
#         for i in range(norb):
#             signa = aindex[:,i,1]
#             signb = bindex[:,i,1]
#             maska = numpy.where(signa!=0)[0]
#             maskb = numpy.where(signb!=0)[0]
#             addra = aindex[maska,i,0]
#             addrb = bindex[maskb,i,0]
#             citmp = lib.take_2d(ci_coeff, addra, addrb)
#             citmp *= signa[maska].reshape(-1,1)
#             citmp *= signb[maskb]
#             #: t1[addra.reshape(-1,1),addrb] += citmp
#             lib.takebak_2d(t1, citmp, maska, maskb)
#         return t1
        
#     ci1 = trans(fcivec, acre, bdes, neleca + 1, nelecb - 1)
#     return ci1 / (numpy.sqrt(s2 / 2 * (s2 / 2 + 1) - ms2 / 2 * (ms2 / 2 + 1)))

def contract_spin_up(fcivec, norb, nelec, s2, ms2):
    neleca, nelecb = fci.addons._unpack_nelec(nelec)
    strsa = fci.cistring.make_strings(range(norb), neleca)
    strsb = fci.cistring.make_strings(range(norb), nelecb)
    na = len(strsa)
    nb = len(strsb)
    fcivec = fcivec.reshape(na,nb)
    def gen_map(fstr_index, strs, nelec, des=True):
        a_index = fstr_index(strs, norb, nelec)
        amap = numpy.zeros((a_index.shape[0],norb,2), dtype=numpy.int32)
        if des:
            for k, tab in enumerate(a_index):
                sign = tab[:,3]
                tab = tab[sign!=0]
                amap[k,tab[:,1]] = tab[:,2:]
        else:
            for k, tab in enumerate(a_index):
                sign = tab[:,3]
                tab = tab[sign!=0]
                amap[k,tab[:,0]] = tab[:,2:]
        return amap
    
    if neleca > 0:
        ades = gen_map(selected_ci.gen_des_linkstr, strsa, neleca)
    else:
        ades = None

    if nelecb > 0:
        bdes = gen_map(selected_ci.gen_des_linkstr, strsb, nelecb)
    else:
        bdes = None

    if neleca < norb:
        acre = gen_map(selected_ci.gen_cre_linkstr, strsa, neleca, False)
    else:
        acre = None

    if nelecb < norb:
        bcre = gen_map(selected_ci.gen_cre_linkstr, strsb, nelecb, False)
    else:
        bcre = None
    def trans(ci_coeff, aindex, bindex):
        if aindex is None or bindex is None:
            return None

        ma = len(aindex)
        mb = len(bindex)
        t1 = numpy.zeros((ma,mb))
        for i in range(norb):
            signa = aindex[:,i,1]
            signb = bindex[:,i,1]
            maska = numpy.where(signa!=0)[0]
            maskb = numpy.where(signb!=0)[0]
            addra = aindex[maska,i,0]
            addrb = bindex[maskb,i,0]
            citmp = lib.take_2d(ci_coeff, addra, addrb)
            citmp *= signa[maska].reshape(-1,1)
            citmp *= signb[maskb]
            #: t1[addra.reshape(-1,1),addrb] += citmp
            lib.takebak_2d(t1, citmp, maska, maskb)
        return t1
        
    cinew = trans(fcivec, acre, bdes)
    return cinew / (numpy.sqrt(s2 / 2 * (s2 / 2 + 1) - ms2 / 2 * (ms2 / 2 + 1)))


def ci_gr_coreex_by_diag(sfnoci, nras1, nras2, nreleras2, num_gr, num_ex, h1eff,eri,po_list,group, ov_list,ecore_list):
    nelecas = sfnoci.nelecas
    neleca, nelecb = nreleras2
    ncas = nras1 + nras2
    ms2 = neleca - nelecb

    rasstring0a = make_non_excited_string(nras1, nras2, neleca)
    rasstring0b = make_non_excited_string(nras1, nras2, nelecb)
    hamiltonian = construct_hamiltonian_matrix((nras1 + neleca, nras1 + nelecb), ncas, h1eff,eri,po_list,group, ov_list,ecore_list)

    n0a = len(rasstring0a)
    n0b = len(rasstring0b)
    e, ci = numpy.linalg.eigh(hamiltonian)
    index = numpy.argsort(e)
    e = e[index]
    ci = ci[:,index]

    ci_gr = []
    ci_ex = []
    for i in range(num_gr):
        cci = ci[:,i]
        ss, multi = spin_op.spin_square0(ci[:,i], ncas, (nras1 + neleca, nras1 + nelecb))
        if round(multi) == 1:
            ci_gr.append((nras1 + neleca, nras1 + nelecb, round(multi)-1, ms2, e[i], cci))
        elif round(multi) == 3:
                    #uci = get_spin_up_state(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb))
            uci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1 , ms2)
            ci_gr.append((nras1 + neleca + 1, nras1 + nelecb - 1, round(multi)-1, ms2 + 2, e[i], uci))
                    #dci = get_spin_down_state(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb))
                    #self.ci.append((nras1+ neleca - 1, nras1 + nelecb +1, round(multi)-1, ms2 - 2, e[i], dci))
        elif round(multi) == 5:
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1, ms2)
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca + 1, nras1 + nelecb - 1), round(multi)-1, ms2 + 2)
            ci_gr.append((nras1 + neleca + 2, nras1 + nelecb - 2, round(multi)-1, ms2 + 4, e[i], cci))

        elif round(multi) == 2:
            ci_gr.append((nras1 + neleca, nras1 + nelecb, round(multi)-1, ms2, e[i], cci))
        elif round(multi) == 4:
            uci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1 , ms2)
            ci_gr.append((nras1 + neleca + 1, nras1 + nelecb - 1, round(multi)-1, ms2 + 2, e[i], uci))
        elif round(multi) == 6:
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1, ms2)
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca + 1, nras1 + nelecb - 1), round(multi)-1, ms2 + 2)
            ci_gr.append((nras1 + neleca + 2, nras1 + nelecb - 2, round(multi)-1, ms2 + 4, e[i], cci))
    e = e[n0a*n0b:]
    ci = ci[:,n0a*n0b:]
    for i in range(num_ex):
        cci = ci[:,i]
        ss, multi = spin_op.spin_square0(ci[:,i], ncas, (nras1 + neleca, nras1 + nelecb))
        if round(multi) == 1:
            ci_ex.append((nras1 + neleca, nras1 + nelecb, round(multi)-1, ms2, e[i], cci))
        elif round(multi) == 3:
                    #uci = get_spin_up_state(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb))
            uci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1 , ms2)
            ci_ex.append((nras1 + neleca + 1, nras1 + nelecb - 1, round(multi)-1, ms2 + 2, e[i], uci))
                    #dci = get_spin_down_state(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb))
                    #self.ci.append((nras1+ neleca - 1, nras1 + nelecb +1, round(multi)-1, ms2 - 2, e[i], dci))
        elif round(multi) == 5:
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1, ms2)
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca + 1, nras1 + nelecb - 1), round(multi)-1, ms2 + 2)
            ci_ex.append((nras1 + neleca + 2, nras1 + nelecb - 2, round(multi)-1, ms2 + 4, e[i], cci))

        elif round(multi) == 2:
            ci_ex.append((nras1 + neleca, nras1 + nelecb, round(multi)-1, ms2, e[i], cci))
        elif round(multi) == 4:
            uci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1 , ms2)
            ci_ex.append((nras1 + neleca + 1, nras1 + nelecb - 1, round(multi)-1, ms2 + 2, e[i], uci))
        elif round(multi) == 6:
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca, nras1 + nelecb), round(multi)-1, ms2)
            cci = contract_spin_up(cci, nras1 + nras2, (nras1 + neleca + 1, nras1 + nelecb - 1), round(multi)-1, ms2 + 2)
            ci_ex.append((nras1 + neleca + 2, nras1 + nelecb - 2, round(multi)-1, ms2 + 4, e[i], cci))
    return ci_gr, ci_ex

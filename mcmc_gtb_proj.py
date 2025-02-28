#!/usr/bin/env python

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import numpy as np
from scipy import linalg 
from numpy import random
import gigrnd


def mcmc(threshold, a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, batch_iteration, out_dir, beta_std, write_psi, write_pst, seed):
    #print('... MCMC ...')

    # seed
    if seed != None:
        random.seed(seed)

    # derived stats
    beta_mrg = np.array(sst_dict['BETA'], ndmin=2).T
    maf = np.array(sst_dict['MAF'], ndmin=2).T
    n_pst = int((n_iter-n_burnin)/thin)
    p = len(sst_dict['SNP'])
    n_blk = len(ld_blk)

    # initialization
    beta = np.zeros((p,1))
    psi = np.ones((p,1))
    sigma = 1.0
    
    if phi == None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    if write_pst == 'TRUE':
        beta_pst = np.zeros((p,n_pst))

    beta_est = np.zeros((p,1))
    psi_est = np.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0
    mm = 0
    for kk in range(n_blk):
        if blk_size[kk] == 0:
            continue
        else:
            idx_blk = range(mm, mm + blk_size[kk])
            eigenval, eigenvec = np.linalg.eigh(ld_blk[kk])
            start_idx = max(sum(eigenval < 0.01), int(np.floor(eigenval.shape[0] * 0)))
            eigenval_new = np.copy(eigenval[start_idx:]); eigenvec_new = np.copy(eigenvec[:, start_idx:])
            ld_blk[kk] = eigenvec_new.dot(np.diag(eigenval_new)).dot(eigenvec_new.T)
            beta_mrg[idx_blk] = eigenvec[:, start_idx:].dot(eigenvec[:, start_idx:].T).dot(beta_mrg[idx_blk])
            mm += blk_size[kk]

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):
        #if itr % 100 == 0:
        #    print('--- iter-' + str(itr) + ' ---')

        mm = 0; quad = 0.0
        for kk in range(n_blk):
            if blk_size[kk] == 0:
                continue
            else:
                idx_blk = range(mm,mm+blk_size[kk])
                dinvt = ld_blk[kk]+np.diag(1.0/psi[idx_blk].T[0])
                dinvt_chol = linalg.cholesky(dinvt)
                beta_tmp = linalg.solve_triangular(dinvt_chol, beta_mrg[idx_blk], trans='T') + np.sqrt(sigma/n)*random.randn(len(idx_blk),1)
                beta[idx_blk] = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N')
                quad += np.dot(np.dot(beta[idx_blk].T, dinvt), beta[idx_blk])
                mm += blk_size[kk]

        err = max(n/2.0*(1.0-2.0*sum(beta*beta_mrg)+quad), n/2.0*sum(beta**2/psi))
        sigma = 1.0/random.gamma((n+p)/2.0, 1.0/err)

        delta = random.gamma(a+b, 1.0/(psi+phi))

        for jj in range(p):
            psi[jj] = gigrnd.gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)
        #psi[psi>1] = 1.0

        if phi_updt == True:
            w = random.gamma(1.0, 1.0/(phi+1.0))
            phi = random.gamma(p*b+0.5, 1.0/(sum(delta)+w))

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1

    # convert standardized beta to per-allele beta
    beta_est[abs(beta_est[:,0]) < threshold, 0] = 0
    active_SNP = []; beta_pst_std = []
    for snp, beta in zip(sst_dict['SNP'], beta_est):
        if beta == 0:
            continue
        active_SNP.append(snp)
        beta_pst_std.append(beta[0])

    if beta_std == 'FALSE':
        beta_est /= np.sqrt(2.0*maf*(1.0-maf))

        if write_pst == 'TRUE':
            beta_pst /= np.sqrt(2.0*maf*(1.0-maf))


    # write posterior effect sizes
    if phi_updt == True:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_iteration%d.txt' % (a, b, batch_iteration)
    else:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_iteration%d.txt' % (a, b, phi, batch_iteration)

    with open(eff_file, 'w') as ff:
        if write_pst == 'TRUE':
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_pst):
                if beta == 0:
                    continue
                ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (batch_iteration, snp, bp, a1, a2, beta))
        else:
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
                if beta == 0:
                    continue
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (batch_iteration, snp, bp, a1, a2, beta))

    # write posterior estimates of psi
    if write_psi == 'TRUE':
        if phi_updt == True:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, batch_iteration)
        else:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, batch_iteration)

        with open(psi_file, 'w') as ff:
            for snp, psi in zip(sst_dict['SNP'], psi_est):
                ff.write('%s\t%.6e\n' % (snp, psi))

    return active_SNP, beta_pst_std



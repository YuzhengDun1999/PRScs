#!/usr/bin/env python

"""
PRS-CS: a polygenic prediction method that infers posterior SNP effect sizes under continuous shrinkage (CS) priors
using GWAS summary statistics and an external LD reference panel.

Reference: T Ge, CY Chen, Y Ni, YCA Feng, JW Smoller. Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors.
           Nature Communications, 10:1776, 2019.


Usage:
python PRScs.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE --n_gwas=GWAS_SAMPLE_SIZE --out_dir=OUTPUT_DIR
                [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN --thin=MCMC_THINNING_FACTOR
                 --chrom=CHROM --write_psi=WRITE_PSI --write_pst=WRITE_POSTERIOR_SAMPLES --seed=SEED]

"""


import os
import sys
import getopt

import parse_genet
import mcmc_gtb_proj
import gigrnd
import pandas as pd

def parse_param():
    long_opts_list = ['ref_dir=', 'bim_prefix=', 'sst_file=', 'a=', 'b=', 'phi=', 'n_gwas=',
                      'n_iter=', 'n_burnin=', 'thin=', 'out_dir=', 'beta_std=', 'write_psi=', 'write_pst=', 'seed=',
                      'iteration=', 'threshold=', 'help']

    param_dict = {'ref_dir': None, 'bim_prefix': None, 'sst_file': None, 'a': 1, 'b': 0.5, 'phi': None, 'n_gwas': None,
                  'n_iter': 1000, 'n_burnin': 500, 'thin': 5, 'out_dir': None, 'chrom': range(1,23),
                  'beta_std': 'FALSE', 'write_psi': 'FALSE', 'write_pst': 'FALSE', 'seed': None,
                  'iteration': 50, 'threshold': None}

    #print('\n')

    if len(sys.argv) > 1:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "h", long_opts_list)          
        except:
            print('Option not recognized.')
            print('Use --help for usage information.\n')
            sys.exit(2)

        for opt, arg in opts:
            if opt == "-h" or opt == "--help":
                print(__doc__)
                sys.exit(0)
            elif opt == "--ref_dir": param_dict['ref_dir'] = arg
            elif opt == "--bim_prefix": param_dict['bim_prefix'] = arg
            elif opt == "--sst_file": param_dict['sst_file'] = arg
            elif opt == "--a": param_dict['a'] = float(arg)
            elif opt == "--b": param_dict['b'] = float(arg)
            elif opt == "--phi": param_dict['phi'] = float(arg)
            elif opt == "--n_gwas": param_dict['n_gwas'] = int(arg)
            elif opt == "--n_iter": param_dict['n_iter'] = int(arg)
            elif opt == "--n_burnin": param_dict['n_burnin'] = int(arg)
            elif opt == "--iteration": param_dict['iteration'] = int(arg)
            elif opt == "--threshold": param_dict['threshold'] = float(arg)
            elif opt == "--thin": param_dict['thin'] = int(arg)
            elif opt == "--out_dir": param_dict['out_dir'] = arg
            elif opt == "--beta_std": param_dict['beta_std'] = arg.upper()
            elif opt == "--write_psi": param_dict['write_psi'] = arg.upper()
            elif opt == "--write_pst": param_dict['write_pst'] = arg.upper()
            elif opt == "--seed": param_dict['seed'] = int(arg)
    else:
        print(__doc__)
        sys.exit(0)

    if param_dict['ref_dir'] == None:
        print('* Please specify the directory to the reference panel using --ref_dir\n')
        sys.exit(2)
    elif param_dict['bim_prefix'] == None:
        print('* Please specify the directory and prefix of the bim file for the target dataset using --bim_prefix\n')
        sys.exit(2)
    elif param_dict['sst_file'] == None:
        print('* Please specify the summary statistics file using --sst_file\n')
        sys.exit(2)
    elif param_dict['n_gwas'] == None:
        print('* Please specify the sample size of the GWAS using --n_gwas\n')
        sys.exit(2)
    elif param_dict['out_dir'] == None:
        print('* Please specify the output directory using --out_dir\n')
        sys.exit(2)

    for key in param_dict:
        print('--%s=%s' % (key, param_dict[key]))

    print('\n')
    return param_dict


def main():
    param_dict = parse_param()

    sst_dict_all = {'CHR':[], 'SNP':[], 'BP':[], 'A1':[], 'A2':[], 'MAF':[], 'BETA':[], 'FLP':[]}
    for chrom in range(1,23):
        if '1kg' in os.path.basename(param_dict['ref_dir']):
            ref_dict = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_1kg_hm3', int(chrom))
        elif 'ukbb' in os.path.basename(param_dict['ref_dir']):
            ref_dict = parse_genet.parse_ref(param_dict['ref_dir'] + '/snpinfo_ukbb_hm3', int(chrom))

        vld_dict = parse_genet.parse_bim(param_dict['bim_prefix'], int(chrom))

        sst_dict = parse_genet.parse_sumstats(ref_dict, vld_dict, param_dict['sst_file'], param_dict['n_gwas'])
        sst_dict_all['CHR'].extend(sst_dict['CHR'])
        sst_dict_all['SNP'].extend(sst_dict['SNP'])
        sst_dict_all['BP'].extend(sst_dict['BP'])
        sst_dict_all['A1'].extend(sst_dict['A1'])
        sst_dict_all['A2'].extend(sst_dict['A2'])
        sst_dict_all['MAF'].extend(sst_dict['MAF'])
        sst_dict_all['BETA'].extend(sst_dict['BETA'])
        sst_dict_all['FLP'].extend(sst_dict['FLP'])

    sumstat_df = pd.DataFrame(sst_dict_all)
    resid_cor = sumstat_df[['SNP', 'BETA']]; resid_cor = resid_cor.copy()
    resid_cor.loc[:, 'cor'] = resid_cor['BETA'].abs()
    ever_active_SNP = []; candidate_SNP = []
    for batch_screening_iter in range(param_dict['iteration']):
        resid_cor = resid_cor[~resid_cor['SNP'].isin(ever_active_SNP)]
        resid_cor_sorted = resid_cor.sort_values(by="cor", ascending=False)
        resid_cor_sorted = resid_cor_sorted.head(1000)
        SNP_selected = resid_cor_sorted['SNP']
        if set(SNP_selected.tolist()).issubset(set(candidate_SNP)):
            break
        candidate_SNP = SNP_selected.tolist() + ever_active_SNP
        candidate_SNP = list(set(candidate_SNP))
        sumstat_candidate = sumstat_df[sumstat_df['SNP'].isin(candidate_SNP)].to_dict(orient='list')
        ld_blk_all = []; blk_size_all = []
        for chrom in range(1, 23):
            ld_blk, blk_size = parse_genet.parse_ldblk(param_dict['ref_dir'], sumstat_candidate, int(chrom))
            ld_blk_all = ld_blk_all + ld_blk
            blk_size_all = blk_size_all + blk_size

        active_SNP, beta_pst_std = mcmc_gtb.mcmc(param_dict['threshold'], param_dict['a'], param_dict['b'], param_dict['phi'], sumstat_candidate, param_dict['n_gwas'], ld_blk_all,
                                                 blk_size_all, param_dict['n_iter'], param_dict['n_burnin'], param_dict['thin'], batch_screening_iter, param_dict['out_dir'],
                                                 param_dict['beta_std'], param_dict['write_psi'], param_dict['write_pst'], param_dict['seed'])
        ever_active_SNP = ever_active_SNP + active_SNP
        ever_active_SNP = list(set(ever_active_SNP))
        resid_cor = parse_genet.get_resid_cor(param_dict['ref_dir'], sumstat_df, pd.DataFrame({'SNP':active_SNP, 'BETA_pst':beta_pst_std}))


if __name__ == '__main__':
    main()



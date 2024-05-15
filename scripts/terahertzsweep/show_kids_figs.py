#!/usr/bin/env python
import sys

def main(argv=None):
    parser = argparse.ArgumentParser(formatter_class=myHelpFormatter, description=__doc__)
    parser.add_argument('--all', action='store_true',
                        help='plot Q, responsivity, tau_qp, NEP figures')
    parser.add_argument('--q', action='store_true',
                        help='plot Q')
#    parser.add_argument('--resp', action='store_true',
#                        help='plot responsivity')
#    parser.add_argument('--tau', action='store_true',
#                        help='plot tau_qp')
#    parser.add_argument('--nep', action='store_true',
#                        help='plot NEP')
    parser.add_argument('--good', action='store_true',
                        help='load disabled_kid_file')
    parser.add_argument('--save', action='store_true',
                        help='save figures to out directory')
    args = parser.parse_args(argv)

#    args_ar = [args.all, args.q, args.resp, args.tau, args.nep]
    args_ar = [args.all, args.q]
    if not (True in args_ar):
        parser.print_help()

    if args.good:
        disabled_kids = []
        with open(disabled_kid_file) as f:
            for l in f:
                l = l.strip()
                if not l or (l[0] == '#'): continue
                if l in disabled_kids: continue
                disabled_kids.append(int(l))
        disabled_kids.sort()
    
    from util import weighted_avg_and_std, delete_elements
    if args.all or args.q:
        ifile = outdir + '/FitSweep_fit.npy'
        print( '== plotting Q values from: %s...' %ifile )
        (kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi) = np.load(ifile)

#        from util import bad_index_by_fraction
#        idx_ar = bad_index_by_fraction(Qr, dQr, thre0=6., thre1=5.)
#        print 'by Qr: ', kidid[idx_ar]
#        idx_ar = bad_index_by_fraction(Qc, dQc, thre0=6., thre1=5.)
#        print 'by Qc: ', kidid[idx_ar]
#        idx_ar = bad_index_by_fraction(Qi, dQi, thre0=6., thre1=5.)
#        print 'by Qi: ', kidid[idx_ar]

        if args.good:
            idx_ar = []
            for bad in disabled_kids:
                if bad in kidid:
                    idx = np.argmin( abs(kidid-bad) )
                    idx_ar.append(idx)
            print( '%d elements are removed by disabled_kid_file..' %len(idx_ar) )
            kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi\
                = delete_elements([kidid, fr, dfr, Qr, dQr, Qc, dQc, Qi, dQi], idx_ar)
        
        fig = plt.figure(figsize=(16,9))
        plt.subplot(231)
        plt.errorbar(kidid, Qr, dQr, color='b', fmt='.')
        plt.grid()
        plt.title('loaded Q')
        plt.xlabel('KID ID')
        plt.ylabel('Qr')
        plt.ylim(1e+3, 1e+7)
        plt.yscale('log')
        
        plt.subplot(232)
        plt.errorbar(kidid, Qc, dQc, color='b', fmt='.')
        plt.grid()
        plt.title('coupling Q')
        plt.xlabel('KID ID')
        plt.ylabel('Qc')
        plt.ylim(1e+3, 1e+7)
        plt.yscale('log')

        plt.subplot(233)
        plt.errorbar(kidid, Qi, dQi, color='b', fmt='.')
        plt.grid()
        plt.title('internal Q')
        plt.xlabel('KID ID')
        plt.ylabel('Qi')
        plt.ylim(1e+3, 1e+7)
        plt.yscale('log')

        plt.subplot(234)
        plt.plot(kidid, dQr/Qr, 'b.')
        plt.grid()
        plt.xlabel('KID ID')
        plt.ylabel('Error fraction of Qr')

        plt.subplot(235)
        plt.plot(kidid, dQc/Qc, 'b.')
        plt.grid()
        plt.xlabel('KID ID')
        plt.ylabel('Error fraction of Qc')

        plt.subplot(236)
        plt.plot(kidid, dQi/Qi, 'b.')
        plt.grid()
        plt.xlabel('KID ID')
        plt.ylabel('Error fraction of Qi')

        if args.save: plt.savefig(os.path.join(outdir, 'figQvalues.png'))

        fig = plt.figure(figsize=(16,5))
        ax = plt.subplot(131)
        mean, std = weighted_avg_and_std(Qr, np.ones(len(Qr)))
        #mean, std = weighted_avg_and_std(Qr, (Qr/dQr)**2)
        plt.hist(Qr, bins=50, alpha=0.5)
        plt.grid()
        plt.title( '%d KIDs, mean: %.2e, std: %.2e' %(len(Qr), mean, std) )
        plt.xlabel('loaded Q')
        #plt.xlim(1e+4, 1e+8)
        #plt.xscale('log')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        ax = plt.subplot(132)
        mean, std = weighted_avg_and_std(Qc, np.ones(len(Qc)))
        #mean, std = weighted_avg_and_std(Qc, (Qc/dQc)**2)
        plt.hist(Qc, bins=50, alpha=0.5)
        plt.grid()
        plt.title( '%d KIDs, mean: %.2e, std: %.2e' %(len(Qc), mean, std) )
        plt.xlabel('coupling Q')
        #plt.xlim(1e+4, 1e+8)
        #plt.xscale('log')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        ax = plt.subplot(133)
        mean, std = weighted_avg_and_std(Qi, np.ones(len(Qi)))
        #mean, std = weighted_avg_and_std(Qi, (Qi/dQi)**2)
        plt.hist(Qi, bins=50, alpha=0.5)
        plt.grid()
        plt.title( '%d KIDs, mean: %.2e, std: %.2e' %(len(Qi), mean, std) )
        plt.xlabel('internal Q')
        #plt.xlim(1e+4, 1e+8)
        #plt.xscale('log')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        if args.save: plt.savefig(os.path.join(outdir, 'figQvalues_hist.png'))

    plt.show()

if __name__ == '__main__':
    sys.path.append('../libs')
    from common import *

    main()


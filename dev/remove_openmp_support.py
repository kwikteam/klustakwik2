import os
import re

fname = '../klustakwik2/numerics/cylib/e_step_cy.pyx'
txt = open(fname, 'r').read()
txt = re.sub('# START_OPEN_MP(.*?)# END_OPEN_MP', '', txt, flags=re.MULTILINE|re.DOTALL)
open(fname, 'w').write(txt)

import sys
def init_opts(opts):
	# system floating point precision
	eps =  sys.float_info.epsilon
	"""
		opt is a opt object
	"""

	# Default values
	DEFAULT_MAX_ITERATION = 1000
	DEFAULT_TOLERANCE     = 1e-10
	MINIMUM_TOLERANCE     = eps * 100
	DEFAULT_TERMINATION_COND = 1
	DEFAULT_INIT = 0
	DEFAULT_PARALLEL_SWITCH = False

	# starting point
	if hasattr(opts, 'init'):
		if(opts.init!=0) and (opts.init!=1) and (opts.init!=2):
			opts.init=DEFAULT_INIT
		if (not hasattr(opts, 'W0')) and (opts.init==1):
			opts.init=DEFAULT_INIT
	else:
		opts.init = DEFAULT_INIT

	# tolerance
	if hasattr(opts, 'tol'):
		# detect if the tolerance is smaller than minimum tolerance allowed.
		if(opts.tol < MINIMUM_TOLERANCE):
			opts.tol = MINIMUM_TOLERANCE
	else:
		opts.tol = DEFAULT_TOLERANCE
    
	# Maximum iteration steps
	if hasattr(opts, 'maxIter'):
	    if (opts.maxIter < 1):
	        opts.maxIter = DEFAULT_MAX_ITERATION
	else:
	    opts.maxIter = DEFAULT_MAX_ITERATION

	# Termination condition
	if hasattr(opts, 'tFlag'):
	    if opts.tFlag < 0:
	        opts.tFlag = 0
	    elif opts.tFlag > 3:
	        opts.tFlag = 3
	    else:
	        opts.tFlag = int(opts.tFlag)
	else:
	    opts.tFlag = DEFAULT_TERMINATION_COND

	# Parallel Options: latter

	# if hasattr(opts, 'pFlag'):
	#     if opts.pFlag == true and not exist(mstring('matlabpool'), mstring('file')):
	#         opts.pFlag = false
	#         warning(mstring('MALSAR:PARALLEL'), mstring('Parallel Toolbox is not detected, MALSAR is forced to turn off pFlag.'))
	#     elif opts.pFlag != true and opts.pFlag != false:
	#         # validate the pFlag.
	#         opts.pFlag = DEFAULT_PARALLEL_SWITCH
	#     end
	# else:
	#     # if not set the pFlag to default.
	#     opts.pFlag = DEFAULT_PARALLEL_SWITCH
	# end

	# if opts.pFlag:
	#     # if the pFlag is checked,
	#     # check segmentation number.
	#     if hasattr(opts, 'pSeg_num'):
	#         if opts.pSeg_num < 0:
	#             opts.pSeg_num = matlabpool(mstring('size'))
	#         else:
	#             opts.pSeg_num = ceil(opts.pSeg_num)
	#         end
	#     else:
	#         opts.pSeg_num = matlabpool(mstring('size'))
	#     end
	# end

	return opts
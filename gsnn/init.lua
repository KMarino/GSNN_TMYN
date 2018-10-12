-- Graph Search Neural Network model.
-- It's a working title
--
-- Kenneth Marino, 06/2016
--

require 'nn'
require 'nngraph'
require 'torch'
require '../ggnn'
require '../graph'

gsnn = {}

gsnn.use_gpu = false

gsnn.IMPORTANCE_NET_PREFIX = 'importance'
gsnn.CONTEXT_NET_PREFIX = 'context'
gsnn.NODE_BIAS_CONTEXT_PREFIX = 'context-bias-input'
gsnn.NODE_BIAS_IMPORTANCE_PREFIX = 'importance-bias-input'
gsnn.NODE_BIAS_ATTENTION_PREFIX = 'attention-bias-input'
gsnn.CONTEXT_GATE_NET_PREFIX = 'context-gate'
gsnn.IMPORTANCE_GATE_NET_PREFIX = 'importance-gate'
gsnn.EDGE_BIAS_PREFIX = 'edge-bias-input'
gsnn.EDGE_BIAS_FORWARD_PREFIX = 'edge-bias-forward-input'
gsnn.EDGE_BIAS_BACKWARD_PREFIX = 'edge-bias-backward-input'
gsnn.ATTENTION_NET_PREFIX = 'attention-net'
gsnn.ATTENTION_NET_FORWARD_PREFIX = 'attention-net-forward'
gsnn.ATTENTION_NET_BACKWARD_PREFIX = 'attention-net-backward'

-- Include files
include('GSNN.lua')
include('gsnn_util.lua')

return gsnn


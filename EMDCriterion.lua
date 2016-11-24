require 'nn'

if not wasserstein then

wasserstein = true;

local WassersteinCriterion, WassersteinCriterionCriterionParent = torch.class('nn.WassersteinCriterion'   , 'nn.Criterion')


function WassersteinCriterion:__init(...)
	
	WassersteinCriterionCriterionParent.__init(self)
	xlua.require('torch',true)
	xlua.require('nn',true)
	local args = dok.unpack(
	{...},
	'nn.WassersteinCriterion','Initialize the WasserStein Criterion',
	{arg='norm', type ='string', help='normalization methods [sum] or softmax',default="sum"},
	{arg='sinkhorn', type ='boolean', help='force to use Sinhorn criteria',default=false},
	{arg='lambda', type ='number', help='sinkhorn regularizatino',default=3},
	{arg='iter', type ='number', help='maximum sinhorn iterations',default=100},
	{arg='L1', type ='boolean', help='Employs the L1 Projected gradient',default=true},
	{arg='rho', type ='number', help='Regularization parameter', default = 2},
	{arg='diff', type ='boolean', help='Employs differentials instead of optimized functions, for debugging purposes',default=false},
	{arg='eps', type ='number', help='epsilon for the differential estimation', default = 1e-3},
	{arg='M', type ='torch.FloatTensor', help='Metric Space'},
	{arg='A', type ='torch.IntTensor', help='Adjancency Matrix'}
	)

	for x,val in pairs(args) do
		self[x] = val
	end
	
	self.sm = nn.SoftMax()
	
	-- M = nil assumes all edge costs to be 1

	--           | A = nil | A:dim()=1
	-- M = nil   | lin,M=1 | tree,M=1
	-- M:dim()=1 |   lin   |   tree
	-- M:dim()=2 |   FC    | XXXXXXXX
	
end

function WassersteinCriterion:preprocess(P, Q)

	-- Update the full criterion to the type required by P
--	self.tensorType = P:type()
	self:type(P:type())
	
	------ if the target is only the index of a coordinate instead of a weight vector, we  transform the target space accordingly
	if Q:dim()==1 or Q:size(Q:dim())==1 then
	
		local Qv = Q:view(-1,1)
		Q = P:view(-1,P:size(P:dim())):clone():fill(0)
		for i=1,Qv:size(1) do
			Q[i][Qv[i][1]] = 1 
		end
	end

	------ The input vectors are flattened and normalized
	assert(P:nElement()==Q:nElement() and P:size(P:dim())==Q:size(Q:dim()), "WassersteinCriterion: vectors do not match sizes")
	local Pv = P:view(-1,P:size(P:dim()))
	local Pn

	if self.norm=="sum" then
	
		self.scales = Pv:sum(2):add(1e-30)
		Pn = torch.cdiv(Pv, self.scales:expandAs(Pv))
		
	elseif self.norm=="softmax" then
		
		self.scales = torch.ones(Pv:size(1),1):type(P:type())
		Pn = self.sm:forward(Pv)
	else
		
		error("Unsupported norm")
	end
	

	------ The output vectors are flattened and normalized
	local Qv = Q:view(-1,Q:size(Q:dim()))	
	local Qn = torch.cdiv(Qv, Q:sum(2):add(1e-30):expandAs(Qv))

	------ We update the A and M matrix accordingly
	local N  = Pv:size(2)
	
	self.A = self.A or torch.linspace(2, 1+N, N):type(P:type())

	local T  = self.A:size(1)

	self.M = self.M or torch.ones(T):type(P:type())

	if self.sinkhorn and self.M:dim() ~= 2 then
	
	
		local M2 = torch.Tensor(T,T):type(P:type())
		
		M2:fill(1e20)

		for i=1,T do M2[i][i]=0 end
		for i=1,T-1 do 
			M2[i][self.A[i]] = self.M[i]
			M2[self.A[i]][i] = self.M[i] 
		end
		for k=1,T do 
			for i=1,T do 
				for j=1,T do 
					if M2[i][j] > M2[i][k] + M2[k][j] then
						M2[i][j] = M2[i][k] + M2[k][j]
					end
				end
			end
		end
		self.M = M2
	end
	self.sinkhorn = (self.M:dim()==2)
	
	return Pn, Qn
end

function WassersteinCriterion:zeroPad(P, sz) 

	if P:size(2) < sz then

		local P2 = P:clone():resize(P:size(1),sz):fill(0)
		P2[{{},{1,P:size(2)}}]:copy(P)
		return P2
	end
	return P
end

function WassersteinCriterion:fSinkhorn(P,Q)

	local M = self.M
	local K = torch.mul(M, -self.lambda):add(-1):exp()

	P = self:zeroPad(P,M:size(1))
	Q = self:zeroPad(Q,M:size(1))
	
	local sheps = 1e-35
	if P:type()== "torch.DoubleTensor" then sheps = 1e-300 end
	
	local h = P:t()
	local y = Q:t()
	local u = P:t():clone():fill(1)

	local Ktu = K:t()*u
	local yKtu = torch.cdiv(y,Ktu:cmax(sheps))
	local KyKtu = K*yKtu
	local uA = torch.cdiv(h,KyKtu:cmax(sheps))
	local uB = u
	for it = 1,self.iter/2 do
	
		if uB:csub(uA):pow(2):cdiv(uA):sum() < 1e-10 then break end
		torch.mm(Ktu, K:t(), uA)
		torch.cdiv(yKtu, y, Ktu:cmax(sheps))
		torch.mm(KyKtu, K, yKtu)
		torch.cdiv(uB, h,KyKtu:cmax(sheps))
		torch.mm(Ktu, K:t(), uB)
		torch.cdiv(yKtu, y, Ktu:cmax(sheps))
		torch.mm(KyKtu, K, yKtu)
		torch.cdiv(uA, h,KyKtu:cmax(sheps))
		uA:cmin(1/sheps):cmax(sheps)
	end
	local v = torch.cdiv(y,K:t()*uA)
	local f = torch.cmul(uA,torch.cmul(K,M)*v):t():sum(2)
	local g = torch.log(uA):div(self.lambda):t()
	
	local GSum = g:sum(2):div(P:size(2)):view(-1,1):expandAs(g)
	g:csub(GSum)
	
	do
		--local u = uA[{{},{1}}]:clone()
		--print((torch.log(u):div(self.lambda) - (torch.log(u):t()*u:clone():fill(1)):expandAs(K):clone():cdiv(self.lambda*K):clone() * u:clone():fill(1)):t())
	end
	return f, g 
end

function WassersteinCriterion:f(P, Q) 

	if self.sinkhorn then local f,g = self:fSinkhorn(P,Q) return f end

	assert(self.M:size(1) == self.A:size(1), "WassersteinCriterion: metric and adjacency matrixes do not match sizes")
	assert(self.M:size(1) >=      P:size(2), "WassersteinCriterion: metric and source vector do not match sizes")

	local phi = self:zeroPad(P-Q, self.M:size(1))
	for i=1,self.A:size(1)-1 do phi[{{},self.A[i]}]:add(phi[{{},i}]) end

	phi:abs():pow(self.rho)
	return torch.mv(phi,self.M)
end

function WassersteinCriterion:calcdiff(P, Q) 

	local Pv = P:view(-1,P:size(P:dim()))
	local Qv = Q:view(-1,Q:size(Q:dim()))
	local G  = Pv:clone()

	for j = 1,Pv:size(2) do
		local dA = torch.zeros(1, Pv:size(2))
		dA[1][j] = 1
		if self.L1 then dA = dA - 1/Pv:size(2) end
		
		G[{{},{j}}] = self:f(Pv+dA:mul(self.eps):expandAs(Pv),Qv) - self:f(Pv,Qv)
	end
	return G:div(self.eps)
end

function WassersteinCriterion:g(P, Q) 

	if self.diff then return self:calcdiff(P,Q) end
	if self.sinkhorn then local f,g = self:fSinkhorn(P,Q) return g end
	
	assert(self.M:size(1) == self.A:size(1), "WassersteinCriterion: metric and adjacency matrixes do not match sizes")
	assert(self.M:size(1) >=      P:size(2), "WassersteinCriterion: metric and source vector do not match sizes")

	local phi = self:zeroPad(P-Q, self.M:size(1))

	for i=1,self.A:size(1)-1 do phi[{{},self.A[i]}]:add(phi[{{},i}]) end
	local phi2 = phi:clone():abs():add(1e-10):pow(self.rho-2)
	phi:cmul(phi2)

	phi:cmul(self.M:view(-1,self.M:size(1)):expandAs(phi)):mul(self.rho)
	
	if self.PRE == nil then
		self.PRE = torch.zeros(self.A:size(1),P:size(2)):type(P:type())
		for k=1,P:size(2) do
			local parent = k
			while true do
				self.PRE[parent][k] = 1
				if parent == self.A:size(1) then break end 
				parent = self.A[parent]
			end
		end	
		self.NC = torch.zeros(self.A:size(1)):type(P:type())
		for k=1,P:size(2) do
			local parent = k
			while true do
				self.NC[parent] = self.NC[parent] +  1
				if parent == self.A:size(1) then break end 
				parent = self.A[parent]
			end
		end	
		self.NC = self.NC/P:size(2)
	end
	local res = torch.mm(phi,self.PRE)
--	local resSum = res:sum(2):div(P:size(2)):view(-1,1):expandAs(res)
--	print(resSum)
--	print(torch.mv(phi,self.NC):view(-1,1):expandAs(res))
	
	return res:csub(torch.mv(phi,self.NC):view(-1,1):expandAs(res))
end


----------- Output interface -----------

function WassersteinCriterion:updateOutput(input, target) 

	local P,Q = self:preprocess(input, target)	
	self.output = self:f(P,Q):cmul(self.scales):sum()/P:size(1)
	return self.output
end

function WassersteinCriterion:updateGradInput(input, target) 

	local P,Q = self:preprocess(input, target)
	self.gradInput = self:g(P, Q)[{{},{1,input:size(input:dim())}}]:clone()

	if self.norm=="sum" then
--		self.gradInput:cmul(self.scales:expandAs(input:view(-1,input:size(input:dim()))))
	elseif self.norm=="softmax" then
		self.gradInput = self.sm:backward(input:view(-1,input:size(input:dim())), self.gradInput)
	end
	return self.gradInput:div(P:size(1))
end


end 




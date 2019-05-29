require 'nn'

local EMDCriterionBase, EMDCriterionBaseParent = torch.class('nn.EMDCriterionBase'   , 'nn.Criterion')

function EMDCriterionBase:__init(...)
	
	EMDCriterionBaseParent.__init(self)
	xlua.require('torch',true)
	xlua.require('nn',true)
	local args = dok.unpack(
	{...},
	'nn.EMDCriterionBase','Initialize the WasserStein Criterion',
	{arg='norm', type ='string', help='normalization:',default=nil},
	{arg='sinkhorn', type ='boolean', help='force to use Sinkhorn criteria',default=false},
	{arg='lambda', type ='number', help='sinkhorn regularization',default=3},
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
	self.normalize = nn.Normalize(1)
	
	-- M = nil assumes all edge costs to be 1

	--           | A = nil | A:dim()=1
	-- M = nil   | lin,M=1 | tree,M=1
	-- M:dim()=1 |   lin   |   tree
	-- M:dim()=2 |   FC    | XXXXXXXX
	
end

function EMDCriterionBase:preprocess(P, Q)

	-- Update the full criterion to the type required by P
	self:type(P:type())
	
	-- For Compatibility with CrossEntropyCriterion:
	-- if the target is only the index of a coordinate instead of a vector, we transform the target space accordingly
	if Q:dim()==1 or Q:size(Q:dim())==1 then
	
		local Qv = Q:view(-1,1)
		Q = P:view(-1,P:size(P:dim())):clone():fill(0)
		for i=1,Qv:size(1) do
			Q[i][Qv[i][1]] = 1 
		end
	end

	-- Input vectors are flattened and normalized
	assert(P:nElement()==Q:nElement() and P:size(P:dim())==Q:size(Q:dim()), "EMDCriterion: vectors do not match sizes")
	local Pv = P:view(-1,P:size(P:dim()))
	local Pn = Pv
	
	--Pn = torch.cdiv(Pn, P:sum(2):add(1e-30):expandAs(Pv))

	-- Output vectors are flattened and normalized
	local Qv = Q:view(-1,Q:size(Q:dim()))	
	local Qn = torch.cdiv(Qv, Q:sum(2):add(1e-30):expandAs(Qv))

	-- N is the size of the output space
	local N  = Pv:size(2)
		
	-- If adjacency matrix is empty, we assume a histogram
	self.A = self.A or torch.linspace(2, 1+N, N):type(P:type())

	-- T is the size of the tree
	local T  = self.A:size(1)
	assert(T>=N, "Tree smaller than output space") 

	-- If distance matrix is empty, we assume to be ones
	self.M = self.M or torch.ones(T):type(P:type()):div(T)

	-- If we want to use Sinkhorn Distance, we must create a square distance matrix using Floyd algorithm
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
	
	-- If the distance matrix is quare, we must use Sinkhorn to solve it
	self.sinkhorn = (self.M:dim()==2)
	
	return Pn, Qn
end

function EMDCriterionBase:zeroPad(P, sz) 

	if P:size(2) < sz then

		local P2 = P:clone():resize(P:size(1),sz):fill(0)
		P2[{{},{1,P:size(2)}}]:copy(P)
		return P2
	end
	return P
end

function EMDCriterionBase:fSinkhorn(P,Q)

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

function EMDCriterionBase:f(P, Q) 

	if self.sinkhorn then local f,g = self:fSinkhorn(P,Q) return f end

	assert(self.M:size(1) == self.A:size(1), "EMDCriterion: metric and adjacency matrixes do not match sizes")
	assert(self.M:size(1) >=      P:size(2), "EMDCriterion: metric and source vector do not match sizes")

	local phi = self:zeroPad(P-Q, self.M:size(1))
	for i=1,self.A:size(1)-1 do phi[{{},self.A[i]}]:add(phi[{{},i}]) end

	phi:abs():pow(self.rho)
	return torch.mv(phi,self.M)
end

function EMDCriterionBase:calcdiff(P, Q) 

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

function EMDCriterionBase:g(P, Q) 

	if self.diff then return self:calcdiff(P,Q) end
	if self.sinkhorn then local f,g = self:fSinkhorn(P,Q) return g end
	
	assert(self.M:size(1) == self.A:size(1), "EMDCriterion: metric and adjacency matrixes do not match sizes")
	assert(self.M:size(1) >=      P:size(2), "EMDCriterion: metric and source vector do not match sizes")

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


local EMDCriterion, EMDCriterionParent = torch.class('nn.EMDCriterion'   , 'nn.EMDCriterionBase')

function EMDCriterion:__init(...)
	
	EMDCriterionParent.__init(self, {...})
	xlua.require('torch',true)
	xlua.require('nn',true)	
end




function EMDCriterion:updateOutput(input, target) 

	local P,Q = self:preprocess(input, target)	
	if self.norm~=nil then
		P = self.norm:forward(P:clone())
	end
	P = self.normalize:forward(P:clone())

	self.output = self:f(P,Q)
	return self.output
end

function EMDCriterion:updateGradInput(input, target) 

	local P,Q = self:preprocess(input, target)
	if self.norm~=nil then
		P1 = self.norm:forward(P)
	end
	P2 = self.normalize:forward(P1 or P)
	
	self.gradInput = self:g(P2 or P1 or P, Q)[{{},{1,input:size(input:dim())}}]:clone()
	
	print (self.normalize.
	
	self.gradInput = self.normalize:backward(P1 or P,self.gradInput:clone())

	if self.norm~=nil then
		self.gradInput = self.norm:updateGradInput(P, self.gradInput)
	end

	return self.gradInput
end





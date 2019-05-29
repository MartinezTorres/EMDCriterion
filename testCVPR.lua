require 'dp'
require 'nn'
require 'nngraph'
require 'image'
require 'optim'
require 'gnuplot'
local signal = require 'signal.fft'
--local nninit = require 'nninit'

local debugger = require('debugger')

--require "WassersteinCriterion"
require 'EMDCriterion.lua'
--require "SinkhornCriterion"

--[[command line arguments]]--
local opt = lapp[[
   -a,--augment                            perform augmentation
   -d,--data          (default psdThoVsFlow) data folder
   -p,--plot                               plot while training
   -r,--learningRate  (default 0.001)        learning rate, for SGD only
   --cuda                                  use CUDA
   -b,--batchSize     (default 4000)       batch size
   -m,--momentum      (default 0.0)        momentum, for SGD only
   -t,--threads       (default 4)          number of threads
]]

-- fix seed
torch.manualSeed(1)

-- sleep function
local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

-- kill existing gnuplot windows
os.execute("killall gnuplot");


-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

--torch.setdefaulttensortype('torch.FloatTensor')
torch.setdefaulttensortype('torch.DoubleTensor')

------------------

function norm1(A) 

	local As = A:sum(1):add(1e-100):expandAs(A)
	local An = torch.cdiv(A, As)
	return An
end


function norm1Good(AA) 

	return torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA))
end
		
function train(model, criterion, sources, targets)

	local f;
	local parameters,gradParameters = model:getParameters()
	
	-- create closure to evaluate f(X) and df/dX
	local feval = function(x)
		-- just in case:
		collectgarbage()

		-- get new parameters
		if x ~= parameters then
			parameters:copy(x)
		end

		-- reset gradients
		gradParameters:zero()

		-- evaluate function for complete mini batch
		local outputs = model:forward(sources)
		f = criterion:forward(outputs, targets)
		
		--print(f)

		-- estimate df/dW
		local df_do = criterion:backward(outputs, targets)
		model:backward(sources, df_do)
		
		--print(df_do)

		-- return f and df/dX
		return f,gradParameters
	end


	-- Perform SGD step:
	local sgdState = sgdState or {
		learningRate = opt.learningRate,
		momentum = opt.momentum,
		learningRateDecay = 5e-7
	}

	local adamConfig = adamConfig or {
		learningRate = opt.learningRate,
		beta1 = 0.01,
		beta2 = 0.001
	}
	
--	print("sgdState "..sgdState.learningRate)
	
--	optim.sgd(feval, parameters, sgdState)
	optim.adam(feval, parameters, adamConfig)

	--print("Finished train: " .. f)
	return f
end

------------------- EXPERIMENT Nº1: PURE ARTIFICIAL CONVERGENCE --------------------
if false then

	local AA = torch.rand(1,5)
	local BB = torch.rand(1,5)
	
--	AA[1][3]=0;
--	BB[1][3]=0;
	
	AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
	BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

	local evalCrit = nn.WassersteinCriterion{sinkhorn=false,reg=1,L1=true};

	local crit = nn.MultiCriterion()
--	crit:add(nn.MSECriterion())
--	crit:add(nn.WassersteinCriterion{sinkhorn=true})
	crit:add(nn.WassersteinCriterion{sinkhorn=false})
--	crit:add(nn.WassersteinCriterion{sinkhorn=false,reg=1,L1=true})
--	crit:add(nn.EMD2CriterionL1{})

	local lambda = 1024*1024
	
	for n = 1, 20 do
	
--		print(lambda.." "..evalCrit:forward(AA,BB))
		print(lambda.." "..(AA-BB):norm())
--		print(crit:backward(AA,BB))
--		print(crit:backward(AA,BB):mul(10000):sum())
--		AA = AA - crit:backward(AA,BB):mul(10000*0.001);
--		AA = AA - crit:backward(AA,BB):mul(100*0.25);
		local nAA = AA - crit:backward(AA,BB):mul(lambda)
		
--		print(nAA)
--		print(evalCrit:forward(nAA,BB))
		

		while evalCrit:forward(AA,BB) < evalCrit:forward(nAA,BB) do
			lambda = lambda/2
			nAA = AA - crit:backward(AA,BB):mul(lambda)
		end
		AA = nAA
--		gnuplot.plot({"y1",AA[1],'|'},{"y2",BB[1],'|'}, {"g", crit:backward(AA,BB):mul(.1)[1],'|'}) sleep(.125)
		
	end
	os.exit()
end


------------------- EXPERIMENT Nº2: WEIGHTS ON PURE ARTIFICIAL CONVERGENCE (Near shore) ---------
if false then

	local f1 = assert(io.open("artificialNearScore.tex", "w"))
	local f2 = assert(io.open("artificialNearLearn.tex", "w"))
	local f3 = assert(io.open("artificialNearSum.tex", "w"))
	
	local criterions = {
--		nn.MultiCriterion():add(nn.MSECriterion()   ,1.00),
		nn.MultiCriterion():add(nn.EMDCriterion()   ,1.00),
		nn.MultiCriterion():add(nn.EMDCriterionL1() ,1.00),
		nn.MultiCriterion():add(nn.EMD2Criterion()  ,1.00),
		nn.MultiCriterion():add(nn.EMD2CriterionL1(),1.00)
	}

	local A = torch.rand(64,64)
	local B = torch.rand(64,64)

	for crit in list_iter(criterions) do

		local AA = A:clone()
		local BB = B:clone()
		
		AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
		BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

		local evalCrit = nn.EMDCriterion();

		local lambda = 1024*1024
		
		f1:write("\\addplot+[line width=2pt, mark=none] coordinates { ")
		f2:write("\\addplot+[line width=2pt, mark=none] coordinates { ")
		f3:write("\\addplot+[line width=2pt, mark=none] coordinates { ")

		for n = 1, 4000 do
			if n<20 or (n%10 == 0 and n<200) or (n%100 == 0) then 
				print(" "..n.." "..lambda.." "..evalCrit:forward(AA,BB).." "..AA:sum())
				f1:write("("..n..","..evalCrit:forward(AA,BB)..") ")
				f2:write("("..n..","..lambda..") ")
				f3:write("("..n..","..AA:sum()..") ")
			end
			
			while evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda/math.sqrt(2)) ,BB) < evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda) ,BB) do
				lambda = lambda/math.sqrt(2)
			end
			while evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda*math.sqrt(2)) ,BB) < evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda) ,BB) do
				lambda = lambda*math.sqrt(2)
			end

			AA = AA - crit:backward(AA,BB):mul(lambda)
	--		gnuplot.plot({"y1",AA[1],'|'},{"y2",BB[1],'|'}, {"g", crit:backward(AA,BB):mul(.1)[1],'|'})
	--		sleep(.05)
		end
		f1:write("};\n")
		f2:write("};\n")
		f3:write("};\n")
	end
	os.exit()
end


------------------- EXPERIMENT Nº2: WEIGHTS ON PURE ARTIFICIAL CONVERGENCE (Far Far away)---------
if false then

	local f1 = assert(io.open("artificialFarScore.tex", "w"))
	local f2 = assert(io.open("artificialFarLearn.tex", "w"))
	local f3 = assert(io.open("artificialFarSum.tex", "w"))

	
	local criterions = {
--		"MSE", nn.MultiCriterion():add(nn.MSECriterion()   ,1.00),
		nn.MultiCriterion():add(nn.EMDCriterion()   ,1.00),
		nn.MultiCriterion():add(nn.EMDCriterionL1() ,1.00),
		nn.MultiCriterion():add(nn.EMD2Criterion()  ,1.00),
		nn.MultiCriterion():add(nn.EMD2CriterionL1(),1.00)
	}

	local A = torch.rand(64,64)
	local B = torch.rand(64,64)
	
	A:narrow(2,1,32):mul(0)
	B:narrow(2,32,32):mul(0)

	for crit in list_iter(criterions) do

		local AA = A:clone()
		local BB = B:clone()
		
		AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
		BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

		local evalCrit = nn.EMDCriterion();

		local lambda = 1024*1024
		
		f1:write("\\addplot+[line width=2pt, mark=none] coordinates { ")
		f2:write("\\addplot+[line width=2pt, mark=none] coordinates { ")
		f3:write("\\addplot+[line width=2pt, mark=none] coordinates { ")

		for n = 1, 4000 do
			if n<20 or (n%10 == 0 and n<200) or (n%100 == 0) then 
				print(" "..n.." "..lambda.." "..evalCrit:forward(AA,BB).." "..AA:sum())
				f1:write("("..n..","..evalCrit:forward(AA,BB)..") ")
				f2:write("("..n..","..lambda..") ")
				f3:write("("..n..","..AA:sum()..") ")
			end

			while evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda/math.sqrt(2)) ,BB) < evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda) ,BB) do
				lambda = lambda/math.sqrt(2)
			end
			while evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda*math.sqrt(2)) ,BB) < evalCrit:forward( AA - crit:backward(AA,BB):mul(lambda) ,BB) do
				lambda = lambda*math.sqrt(2)
			end
			AA = AA - crit:backward(AA,BB):mul(lambda)
	--		gnuplot.plot({"y1",AA[1],'|'},{"y2",BB[1],'|'}, {"g", crit:backward(AA,BB):mul(.1)[1],'|'})
	--		sleep(.05)
		end
		f1:write("};\n")
		f2:write("};\n")
		f3:write("};\n")
	end
	os.exit()
end

------------------- EXPERIMENT Nº3: THE ARTIFICIAL DISTRIBUTION (FORGET IT) ---------
if false then

	local nFreq = 10
	local nSampxFreq = 10
	local noiseVar = 0.01
	local source = torch.zeros(nFreq,nSampxFreq,64)
	local target = torch.zeros(nFreq,nSampxFreq,source:size(3)/2+1)
	for n1=1,nFreq do
		source[n1] = torch.randn(nSampxFreq,source:size(3)):mul(noiseVar);
		for n2 = 1,nSampxFreq do
			local f = 4+n1
--			print( torch.range(3.1416*2*(n2/nSampxFreq),3.1416*2*(1+n2/nSampxFreq),3.1416*2/(sourceData:size(3)-0.9)):sin():size())
--			print( sourceData[n1][n2]:size() )
			source[n1][n2] = source[n1][n2] + torch.range(3.1416*2*f*(n2/nSampxFreq),3.1416*2*f*(1+n2/nSampxFreq),3.1416*2*f/(source:size(3)-0.99)):sin()

			local targetF = signal.rfft(source[n1][n2])
			target[n1][n2] = torch.pow(targetF[{{},{1}}],2)+torch.pow(targetF[{{},{2}}],2)

--			gnuplot.plot({"y1",source[n1][n2]})
--			gnuplot.plot({"y1",target[n1][n2]})
--			sleep(.2)
		end
	end
	
	local input = nn.Identity()()
	local model = nn.Identity()(input)
	model = nn.View(-1,source:size(3),1):setNumInputDims(2)(model)
	model = nn.TemporalConvolution(1, 16, 5)(model)
	model = nn.Tanh()(model)
	model = nn.Reshape((source:size(3)- 5 + 1)*16,true)(model)
	model = nn.Linear((source:size(3)- 5 + 1)*16, target:size(3))(model)
	model = nn.Square()(model) -- IMPORTANT!!!
	model = nn.gModule({input},{model})

	-------------- DEFINE CRITERION -------------
	criterion = nn.MultiCriterion()
	criterion = criterion:add(nn.MSECriterion(),.5)
	criterion = criterion:add(nn.EMD2CriterionL1(),.5/sqrt(target:size(3)))
	
	for n = 1,100000 do
		
		local sources = source:view(-1,source:size(3))
		local targets = target:view(-1,target:size(3))
		train(model, criterion, sources, targets)
		local G = criterion:backward(model:forward(sources),targets)
		G = G:mul(1/G:norm())
		local GMSE = nn.MSECriterion   ():backward(model:forward(sources),targets)
		GMSE = GMSE:mul(1/GMSE:norm())
		local GEMD = nn.EMD2CriterionL1():backward(model:forward(sources),targets)
		GEMD = GEMD:mul(1/GEMD:norm())
		print("                               "..math.sqrt(GMSE:dot(G)).." "..math.sqrt(GEMD:dot(G)))
		
--		print(nn.MSECriterion():backward(source:view(-1,source:size(3)),target:view(-1,target:size(3))):norm():dot(crit:backward(source:view(-1,source:size(3)),target:view(-1,target:size(3))):norm()))
--		print(nn.MSECriterion()   :backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000):dot(criterion:backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000)))
--		print(nn.EMD2CriterionL1():backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000):dot(criterion:backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000)))
	end
	os.exit()
end

------------------- EXPERIMENT Nº4: THE ARTIFICIAL DISTRIBUTION WITH HOLES (FORGET IT) ---------
if false then

-- source: l=600 sequence with a wave of frequency 20-60 + noise
-- target: the PDF of such sequence

	
	local nFreq = 11
	local nSampxFreq = 1000
	local noiseVar = .2
	local source = torch.zeros(nFreq,nSampxFreq,64)
	local target = torch.zeros(nFreq,nSampxFreq,source:size(3)/2+1)
	for n1=1,nFreq do
		source[n1] = torch.randn(nSampxFreq,source:size(3)):mul(noiseVar);
		for n2 = 1,nSampxFreq do
			local f = 4+n1
--			print( torch.range(3.1416*2*(n2/nSampxFreq),3.1416*2*(1+n2/nSampxFreq),3.1416*2/(sourceData:size(3)-0.9)):sin():size())
--			print( sourceData[n1][n2]:size() )
			source[n1][n2] = source[n1][n2] + torch.range(3.1416*2*f*(n2/nSampxFreq),3.1416*2*f*(1+n2/nSampxFreq),3.1416*2*f/(source:size(3)-0.99)):sin()

			local targetF = signal.rfft(source[n1][n2])
			target[n1][n2] = torch.pow(targetF[{{},{1}}],2)+torch.pow(targetF[{{},{2}}],2)
			target[n1][n2]:mul(1/6000.)

--			gnuplot.plot({"y1",source[n1][n2]})
--			gnuplot.plot({"y1",target[n1][n2]})
--			sleep(.2)
		end
	end
	
	local input = nn.Identity()()
	local model = nn.Identity()(input)
	model = nn.View(-1,source:size(3),1):setNumInputDims(2)(model)
	model = nn.TemporalConvolution(1, 16, 11)(model)
	model = nn.Tanh()(model)
	model = nn.Reshape((source:size(3)- 11 + 1)*16,true)(model)
	model = nn.Linear((source:size(3)- 11 + 1)*16, target:size(3))(model)
	model = nn.Square()(model) -- IMPORTANT!!!
	model = nn.gModule({input},{model})

	-------------- DEFINE CRITERION -------------
	local criterion = nn.MultiCriterion()
	criterion = criterion:add(nn.MSECriterion(),.5)
	criterion = criterion:add(nn.EMD2CriterionL1(),.5*1)

	local evalCriterion = nn.EMDCriterion()
	
	for n = 1,100000 do
		
		local trainSources = torch.zeros(100,source:size(3))
		local trainTargets = torch.zeros(trainSources:size(1),target:size(3))
		
		local testSources = torch.zeros(100,source:size(3))
		local testTargets = torch.zeros(testSources:size(1),target:size(3))
		
		for n2 = 1, trainSources:size(1) do
			local a = 1+2*torch.random(1,nFreq/2)
			local b = torch.random(1,nSampxFreq)
			trainSources[n2] = source[a][b]
			trainTargets[n2] = target[a][b]
			for n3 = 1, target:size(3)-2 do
				trainTargets[n2][n3] = trainTargets[n2][n3]*.25 + trainTargets[n2][n3+1]*.5 + trainTargets[n2][n3+2]*.25
			end
		end

		for n2 = 1, testSources:size(1) do
			local a = 2*torch.random(1,nFreq/2)
			local b = torch.random(1,nSampxFreq)
			testSources[n2] = source[a][b]
			testTargets[n2] = target[a][b]
		end
		
		local f = train(model,criterion,trainSources,trainTargets)
		local G = criterion:backward(model:forward(testSources),testTargets)
		G = G:mul(1/G:norm())
		local GMSE = nn.MSECriterion   ():backward(model:forward(testSources),testTargets)
		GMSE = GMSE:mul(1/GMSE:norm())
		local GEMD = nn.EMD2CriterionL1():backward(model:forward(testSources),testTargets)
		GEMD = GEMD:mul(1/GEMD:norm())
		print("Train: " .. evalCriterion:forward(model:forward(trainSources),trainTargets).." Test: "..evalCriterion:forward(model:forward(testSources),testTargets).."                               "..math.sqrt(GMSE:dot(G)).." "..math.sqrt(GEMD:dot(G)))
		
		if n%100 == 0 then
		gnuplot.figure(1)
		gnuplot.plot({"y1",norm1(model:forward(trainSources[{{1},{}}])[1])},{"y2",norm1(trainTargets[1])})

		gnuplot.figure(2)
		gnuplot.plot({"y1",norm1(model:forward(testSources[{{1},{}}])[1])},{"y2",norm1(testTargets[1])})

		end
--		print(nn.MSECriterion():backward(source:view(-1,source:size(3)),target:view(-1,target:size(3))):norm():dot(crit:backward(source:view(-1,source:size(3)),target:view(-1,target:size(3))):norm()))
--		print(nn.MSECriterion()   :backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000):dot(criterion:backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000)))
--		print(nn.EMD2CriterionL1():backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000):dot(criterion:backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000)))
	end
	os.exit()
end


if false then

	local criterions = {
		{"gMSE",nn.MultiCriterion():add(nn.MSECriterion()   ,1.00), 4.},
		{"gEMD",nn.MultiCriterion():add(nn.EMDCriterion()   ,1.00), .1},
		{"gEMD_L1",nn.MultiCriterion():add(nn.EMDCriterionL1() ,1.00), .1},
		{"gEMD2",nn.MultiCriterion():add(nn.EMD2Criterion()  ,1.00), .1},
		{"gEMD2_L1",nn.MultiCriterion():add(nn.EMD2CriterionL1(),1.00), .1}
	}

--	local criterions = {}
--	criterions["gMSE"] = nn.MultiCriterion():add(nn.MSECriterion()   ,1.00)
--	criterions["gEMD"] = nn.MultiCriterion():add(nn.EMDCriterion()   ,1.00)
--	criterions["gEMD_L1"] = nn.MultiCriterion():add(nn.EMDCriterionL1() ,1.00)
--	criterions["gEMD2"] = nn.MultiCriterion():add(nn.EMD2Criterion()  ,1.00)
--	criterions["gEMD2_L1"] = nn.MultiCriterion():add(nn.EMD2CriterionL1(),1.00)

	local A = torch.rand(1,9)
	local B = torch.rand(1,9)
	
	local AA = A:clone()
	local BB = B:clone()
	
	AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
	BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

	local plots = {{"p",AA[1],'|'},{"q",BB[1],'|'}}
	
	for name,crit in pairs(criterions) do
		table.insert(plots, {crit[1], crit[2]:backward(AA,BB):mul(crit[3])[1],'+-'})
	end
	
	gnuplot.figure(1)
	gnuplot.plot(plots)
	
	local A = torch.ones(1,9)
	local B = torch.ones(1,9)
	
	A[1][3]=10
	B[1][7]=10
	
	local AA = A:clone()
	local BB = B:clone()
	
	AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
	BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

	local plots = {{"p",AA[1],'|'},{"q",BB[1],'|'}}
	
	for name,crit in pairs(criterions) do
		table.insert(plots, {crit[1], crit[2]:backward(AA,BB):mul(crit[3])[1],'+-'})
	end
	
	gnuplot.figure(2)
	gnuplot.plot(plots)
	
		
	local A = torch.ones(1,9)
	local B = torch.ones(1,9)
	
	A[1][5]=10
	
	local AA = A:clone()
	local BB = B:clone()
	
	AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
	BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

	local plots = {{"p",AA[1],'|'},{"q",BB[1],'|'}}
	
	for name,crit in pairs(criterions) do
		table.insert(plots, {crit[1], crit[2]:backward(AA,BB):mul(crit[3])[1],'+-'})
	end
	
	gnuplot.figure(3)
	gnuplot.plot(plots)

	local A = torch.ones(1,15)
	local B = torch.ones(1,15)
	
	A[1][5]=15

	local AA = A:clone()
	local BB = B:clone()
	
	AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
	BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

	local plots = {{"p",AA[1],'|'},{"q",BB[1],'|'}}
	
	for name,crit in pairs(criterions) do
		table.insert(plots, {crit[1], crit[2]:backward(AA,BB):mul(crit[3]*(9/A:size(2)))[1],'+-'})
	end
	
	gnuplot.figure(4)
	gnuplot.plot(plots)
	
	sleep(10)
	os.exit()
end

------------------- EXPERIMENT Nº5: THE ARTIFICIAL DISTRIBUTION WITHOUT HOLES (AGAIN) ---------
if false then

	local criterions = {
		{"gMSE",nn.MultiCriterion():add(nn.MSECriterion()   ,1.00), 4.},
--		{"gEMD",nn.MultiCriterion():add(nn.EMDCriterion()   ,1.00), .1},
--		{"gEMD_L1",nn.MultiCriterion():add(nn.EMDCriterionL1() ,1.00), .1},
--		{"gEMD2",nn.MultiCriterion():add(nn.EMD2Criterion()  ,1.00), .1},
		{"gEMD_L1",nn.MultiCriterion():add(nn.WassersteinCriterion{sinkhorn=false,reg=1},1.00), .1},
		{"gEMD2_L1",nn.MultiCriterion():add(nn.WassersteinCriterion{sinkhorn=false},1.00), .1},
		{"gSinkhorn",nn.MultiCriterion():add(nn.WassersteinCriterion{sinkhorn=true},1.00), .1},
		{"gEMD2_L1_old",nn.MultiCriterion():add(nn.EMD2CriterionL1(),1.00), .1},
	}

	
	local file = assert(io.open("experimentSinus.tex", "w"))
	
-- source: l=600 sequence with a wave of frequency 20-60 + noise
-- target: the PDF of such sequence

	for name,crit in pairs(criterions) do
	
		local criterion = crit[2]

		local nFreq = 11
		local nSampxFreq = 1000
		local noiseVar = .2
		local source = torch.zeros(nFreq,nSampxFreq,64)
		local target = torch.zeros(nFreq,nSampxFreq,source:size(3)/2+1)
		for n1=1,nFreq do
			source[n1] = torch.randn(nSampxFreq,source:size(3)):mul(noiseVar);
			for n2 = 1,nSampxFreq do
				local f = 4+n1
	--			print( torch.range(3.1416*2*(n2/nSampxFreq),3.1416*2*(1+n2/nSampxFreq),3.1416*2/(sourceData:size(3)-0.9)):sin():size())
	--			print( sourceData[n1][n2]:size() )
				source[n1][n2] = source[n1][n2] + torch.range(3.1416*2*f*(n2/nSampxFreq),3.1416*2*f*(1+n2/nSampxFreq),3.1416*2*f/(source:size(3)-0.99)):sin()

				local targetF = signal.rfft(source[n1][n2])
				target[n1][n2] = torch.pow(targetF[{{},{1}}],2)+torch.pow(targetF[{{},{2}}],2)
				target[n1][n2]:mul(1/1000.)

				target[n1][n2] = nn.Normalize(1):forward(target[n1][n2]) -- MSE

	--			target[n1][n2] = nn.LogSoftMax():forward(nn.Normalize(1):forward(target[n1][n2])) LK

	--			gnuplot.plot({"y1",source[n1][n2]})
	--			gnuplot.plot({"y1",target[n1][n2]})
	--			sleep(1)
			end
		end
		
		local input = nn.Identity()()
		local model = nn.Identity()(input)
		model = nn.View(-1,source:size(3),1):setNumInputDims(2)(model)
		model = nn.TemporalConvolution(1, 16, 11)(model)
		model = nn.Tanh()(model)
		model = nn.Reshape((source:size(3)- 11 + 1)*16,true)(model)
		model = nn.Linear((source:size(3)- 11 + 1)*16, target:size(3))(model)
		model = nn.Square()(model) -- IMPORTANT!!!
	--	model = nn.Normalize(1)(model)
	--model = nn.LogSoftMax()(model) KL
		model = nn.gModule({input},{model})

		-------------- DEFINE CRITERION -------------
		local criterion = nn.MultiCriterion()
	--	criterion = criterion:add(nn.DistKLDivCriterion(),1)
	--	criterion = criterion:add(nn.MSECriterion(),.5)
	--	criterion = criterion:add(nn.EMD2CriterionL1(),.5)
	--	criterion = criterion:add(nn.EMDCriterion(),.5)
	--	criterion = criterion:add(nn.EMD2Criterion(),.5)
		criterion = criterion:add(nn.EMDCriterionL1(),.5)

		local function eval(A, B)
			return nn.EMDCriterion():forward(model:forward(A),B)
		end

		local function evalKL(A, B)
			local A1 = model:forward(A):exp()
			local B1 = torch.exp(B)
			return nn.EMDCriterion():forward(A1:add(-torch.min(A1)),B1:add(-torch.min(B1)))
		end
		
		file:write("\\addplot+[line width=2pt, mark=none] coordinates { ")

		for n = 1,2000 do
			
			local trainSources = torch.zeros(100,source:size(3))
			local trainTargets = torch.zeros(trainSources:size(1),target:size(3))
			
			local testSources = torch.zeros(100,source:size(3))
			local testTargets = torch.zeros(testSources:size(1),target:size(3))
			
			for n2 = 1, trainSources:size(1) do
				local a = torch.random(1,nFreq)
				local b = torch.random(1,nSampxFreq)
				trainSources[n2] = source[a][b]
				trainTargets[n2] = target[a][b]
	--			for n3 = 1, target:size(3)-2 do
	--				trainTargets[n2][n3] = trainTargets[n2][n3]*.25 + trainTargets[n2][n3+1]*.5 + trainTargets[n2][n3+2]*.25
	--			end
			end

			for n2 = 1, testSources:size(1) do
				local a = torch.random(1,nFreq)
				local b = torch.random(1,nSampxFreq)
				testSources[n2] = source[a][b]
				testTargets[n2] = target[a][b]
			end
			
			local f = train(model,criterion,trainSources,trainTargets)
			local G = criterion:backward(model:forward(testSources),testTargets)
			G = G:mul(1/G:norm())
			local GMSE = nn.MSECriterion   ():backward(model:forward(testSources),testTargets)
			GMSE = GMSE:mul(1/GMSE:norm())
			local GEMD = nn.EMD2CriterionL1():backward(model:forward(testSources),testTargets)
			GEMD = GEMD:mul(1/GEMD:norm())
			print("Train: " .. eval(trainSources,trainTargets).." Test: "..eval(testSources,testTargets).."                               "..math.sqrt(GMSE:dot(G)).." "..math.sqrt(GEMD:dot(G)))

			if n<20 or (n%10 == 0 and n<200) or (n%100 == 0) then 
				file:write("("..n..","..eval(testSources,testTargets)..") ")
			end
			
			if n%100 == 0 then
			gnuplot.figure(1)
			gnuplot.plot({"y1",norm1(model:forward(trainSources[{{1},{}}])[1])},{"y2",norm1(trainTargets[1])},{"gEMD", nn.EMD2CriterionL1():backward(model:forward(trainSources),trainTargets)[1]},{"gEMD", nn.MSECriterion():backward(model:forward(trainSources),trainTargets)[1]:mul(10000)})

			gnuplot.figure(2)
			gnuplot.plot({"y1",norm1(model:forward(testSources[{{1},{}}])[1])},{"y2",norm1(testTargets[1])})

			end
	--		print(nn.MSECriterion():backward(source:view(-1,source:size(3)),target:view(-1,target:size(3))):norm():dot(crit:backward(source:view(-1,source:size(3)),target:view(-1,target:size(3))):norm()))
	--		print(nn.MSECriterion()   :backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000):dot(criterion:backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000)))
	--		print(nn.EMD2CriterionL1():backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000):dot(criterion:backward(model:forward(source:view(-1,source:size(3))),target:view(-1,target:size(3))):renorm(2,2,0.001):mul(1000)))
		end
		file:write("};\n")
	end

	os.exit()
end

------------------- EXPERIMENT Nº5: REAL DATA ---------
if false then


	local criterions = {
		{"gMSE",nn.MultiCriterion():add(nn.MSECriterion()   ,1.00), .1},
--		{"gEMD",nn.MultiCriterion():add(nn.EMDCriterion()   ,1.00), .1},
--		{"gEMD_L1",nn.MultiCriterion():add(nn.EMDCriterionL1() ,1.00), .1},
--		{"gEMD2",nn.MultiCriterion():add(nn.EMD2Criterion()  ,1.00), .1},
		{"gEMD2_L1",nn.MultiCriterion():add(nn.WassersteinCriterion{sinkhorn=false},1.00), .1},
		{"gSinkhorn",nn.MultiCriterion():add(nn.WassersteinCriterion{sinkhorn=true},1.00), .1},
--		{"gEMD2_L1_old",nn.MultiCriterion():add(nn.EMD2CriterionL1(),1.00), .1},
	}
	
	local fileTest = assert(io.open("experimentPSD_test.tex", "w"))
	local fileTrain = assert(io.open("experimentPSD_train.tex", "w"))
	
-- source: l=600 sequence with a wave of frequency 20-60 + noise
-- target: the PDF of such sequence

	for name,crit in pairs(criterions) do
	
		local criterion = crit[2]

		local testSource  = torch.load(opt.data.."/testSource.th7" ):typeAs(torch.ones(1,1))
		local testTarget  = torch.load(opt.data.."/testTarget.th7" ):narrow(2,1,30):typeAs(torch.ones(1,1))
		local trainSource = torch.load(opt.data.."/trainSource.th7")
		local trainTarget = torch.load(opt.data.."/trainTarget.th7"):narrow(2,1,30)

		testSource  = testSource/testSource:std()
		testTarget  = testTarget/testTarget:std()
		trainSource = trainSource/trainSource:std()
		trainTarget = trainTarget/trainTarget:std()

		local input = nn.Identity()()
		local model = nn.Identity()(input)
		model = nn.View(-1,testSource:size(2),1):setNumInputDims(2)(model)
		model = nn.TemporalConvolution(1, 16, 11)(model)
		model = nn.Tanh()(model)
		model = nn.Reshape((testSource:size(2)- 11 + 1)*16,true)(model)
		model = nn.Linear((testSource:size(2)- 11 + 1)*16, testTarget:size(2))(model)
		model = nn.Square()(model) -- IMPORTANT!!!
	--	model = nn.Normalize(1)(model)
	--model = nn.LogSoftMax()(model) KL
		model = nn.gModule({input},{model})

		local function eval(A, B)
		
--			print(B:size())
			local s = torch.randperm(A:size(1)):long()[{{1,math.min(1200,A:size(1))}}]
			return nn.EMDCriterion():forward(model:forward(A:index(1,s)),B:index(1,s))
		end
		
		fileTest:write("\\addplot+[line width=2pt, mark=none] coordinates { ")
		fileTrain:write("\\addplot+[line width=2pt, mark=none] coordinates { ")

		for n = 1,2000 do
			
			local trainSources = torch.zeros(50,trainSource:size(2))
			local trainTargets = torch.zeros(trainSources:size(1),trainTarget:size(2))
			
			for n2 = 1, trainSources:size(1) do
				local a = torch.random(1,trainSource:size(1))
				trainSources[n2] = trainSource[a]
				trainTargets[n2] = trainTarget[a]
			end
			
			local f = train(model,criterion,trainSources,trainTargets)
			local e = eval(trainSources,trainTargets)
			print("Train: " .. e)
			fileTrain:write("("..n..","..e..") ")

			if n<120 or (n%5 == 0 and n<500) or (n%10 == 0) then 
				local et = eval(testSource,testTarget)
				print("Test: "..et)
				fileTest:write("("..n..","..et..") ")
			end
		end
		fileTest:write("};\n")
		fileTrain:write("};\n")
	end

	os.exit()
end


--------- higher resolution stuff
if false then

	local allCriterions = { 
		gMSE = {
			{"{/Symbol \\321}MSE",nn.MultiCriterion():add(nn.MSECriterion()   ,1.00), 20.},
		},
		gEMD = {
			{"{/Symbol \\321}EMD^1",nn.WassersteinCriterion{rho=1}, .01},
			{"{/Symbol \\321}EMD^2",nn.WassersteinCriterion{rho=2}, .01},
		},
		gSD = {
			{"{/Symbol \\321}SD_{ {/Symbol l} = 0.5}",nn.WassersteinCriterion{sinkhorn=true,lambda=.5}, .01},
			{"{/Symbol \\321}SD_{ {/Symbol l} = 1}",nn.WassersteinCriterion{sinkhorn=true,lambda=1}, .01},
			{"{/Symbol \\321}SD_{ {/Symbol l} = 10}",nn.WassersteinCriterion{sinkhorn=true,lambda=10}, .01},
		},
	}
	for graphname,criterions in pairs(allCriterions) do
	
		local A = torch.ones(1,39)
		local B = torch.ones(1,39)
		
		for i = 1,A:size(2) do
			A[1][i] = math.exp(-((i-7)*(i-7))/(2*8.1))
	--		B[1][i] = math.exp(-((i-30)*(i-30))/(2*20))
	--		B[1][i] = math.exp(-((i-50)*(i-50))/(2*20))
			B[1][i] = math.exp(-((i-33)*(i-33))/(2*8.1))
			if (i%2==0) then B[1][i]=0 end
		end
		
		
		local AA = A:clone()
		local BB = B:clone()
		
		AA = torch.cdiv(AA, AA:clone():sum(2):add(1e-10):expandAs(AA)) -- normalize sum
		BB = torch.cdiv(BB, BB:clone():sum(2):add(1e-10):expandAs(BB)) -- normalize sum

		local plots = {{"p",AA[1],'|'},{"q",BB[1],'|'}}
		
		for name,crit in pairs(criterions) do
			print(crit[1].." "..(crit[2]:forward(AA,BB)*crit[3]*10000))
			table.insert(plots, {crit[1], crit[2]:backward(AA,BB):mul(crit[3])[1],'+-'})
		end
		
		gnuplot.epsfigure("figs/"..graphname.. ".eps")
		gnuplot.figure(2)
		gnuplot.raw("set style fill solid noborder")
		gnuplot.raw("set key off p")
		gnuplot.raw("set key left bottom reverse Left")
		gnuplot.raw("unset autoscale y")
		gnuplot.raw("set yrange [-.4:.4]")
		gnuplot.raw("set size .6 .6")
		gnuplot.plot(plots)
		gnuplot.figprint("figs/"..graphname.. ".eps")
		gnuplot.plotflush()
--		sleep(10)
	end
	
	
	os.exit()
end

-- ANIMATION

if true then

	local criterions = {
		{"gMSE",nn.MultiCriterion():add(nn.MSECriterion()   ,1.00), 4.},
--		{"gEMD",nn.MultiCriterion():add(nn.EMDCriterion()   ,1.00), .05},
		{"gEMD_L1",nn.EMDCriterion{rho=1}, .1},
--		{"gEMD2",nn.MultiCriterion():add(nn.EMD2Criterion()  ,1.00), .1},
		{"gEMD2_L1",nn.EMDCriterion{rho=2}, .1},
		{"gSD",nn.EMDCriterion{sinkhorn=true}, .1},
	}
	
	
--	local name = "gMSE" local crit = nn.MultiCriterion():add(nn.MSECriterion() ,4000.00)
	local name = "gEMD" local crit = nn.MultiCriterion():add(nn.EMDCriterion{sinkhorn=false,rho=5} ,1.00)
--	local name = "gSD" local crit = nn.MultiCriterion():add(nn.EMDCriterion{sinkhorn=true} ,1.00)

	local A = torch.ones(1,40)
	local B = torch.ones(1,40)
	
	local p = 30
	local v = 20
	
	for i = 1,A:size(2) do
		A[1][i] = math.exp(-((i-10)*(i-10))/(2*18))
		B[1][i] = math.exp(-((i-p)*(i-p))/(2*v))
		if (i%2==0) then B[1][i]=0 end
	end
	
	A[1] = A[1]:add(0.0001);
	B[1] = B[1]:add(0.0001);
	
	
	local AA = A:clone()
	local BB = B:clone()
	
	AA = torch.cdiv(AA, AA:sum(2):add(1e-100):expandAs(AA)) -- normalize sum
	BB = torch.cdiv(BB, BB:sum(2):add(1e-100):expandAs(BB)) -- normalize sum

	for it = 100, 200 do
--		local plots = {{"q",BB[1]:clone(),'|'},{"p",AA[1]:clone(),'|'}}
		
--		for name,crit in pairs(criterions) do
			--table.insert(plots, {crit[1], crit[2]:backward(AA,BB):mul(crit[3])[1],'+-'})
			
			for i=1,19 do 
				AA = AA:add(-crit:backward(AA,BB):mul(.00001))
			end
			
--		end

		local plots = {{"q",BB[1]:clone(),'|'},{"p",AA[1]:clone(),'|'}}
		
--		gnuplot.pngfigure("figs/"..name..it..".png")
		gnuplot.title(name)
--		gnuplot.raw("set style fill solid border -1")
		gnuplot.raw("set style fill solid border 0")
		gnuplot.axis({0,41,-0.02,0.20})
		gnuplot.figure(1)
		gnuplot.plot(plots)
		gnuplot.plotflush()

		sleep(.5)
		print(it)
	end
	os.exit()
end

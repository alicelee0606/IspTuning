local cv = require 'cv'
require 'cv.highgui' -- GUI
require 'cv.videoio' -- Video stream
require 'cv.imgproc' -- Image processing (resize, crop, draw text, ...)
require 'cv.imgcodecs'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'unn'

--VideoCapture for accessing the camera
local capture = cv.VideoCapture{device=0}
if not capture:isOpened() then
   print("Failed to open the default camera")
   os.exit(-1)
end
cv.namedWindow{'demo'}

local uvc_params = {
    cv.CAP_PROP_BRIGHTNESS,
    cv.CAP_PROP_CONTRAST,
    cv.CAP_PROP_SATURATION,
    cv.CAP_PROP_HUE,
    cv.CAP_PROP_GAIN,    --not supported in V4L
    cv.CAP_PROP_EXPOSURE --not supported in V4L
}


--Trackbar for adjusting the parameters
local curr_params = {0, 0, 0, 0}
local alphaSliderMax = 100
local alphas = { 0, 0, 0, 0}
for i=1,4 do
    curr_params[i] = capture:get{uvc_params[i]}
    print(curr_params[i])
end

local ffi = require 'ffi'
local alphaSliders = {
    ffi.new('int[1]', curr_params[1]*alphaSliderMax),
    ffi.new('int[1]', curr_params[2]*alphaSliderMax),
    ffi.new('int[1]', curr_params[3]*alphaSliderMax),
    ffi.new('int[1]', curr_params[4]*alphaSliderMax)
}

local function onTrackbar()
    for i=1,4 do
        alphas[i] = alphaSliders[i][0]/alphaSliderMax
        capture:set{uvc_params[i], alphas[i]}
    end
end

local trackbarName = {
    'BRIGHTNESS',
    'CONTRAST  ',
    'SATURATION',
    'HUE       '
}
cv.createTrackbar{trackbarName[1], 'demo', alphaSliders[1], alphaSliderMax, onTrackbar}
cv.createTrackbar{trackbarName[2], 'demo', alphaSliders[2], alphaSliderMax, onTrackbar}
cv.createTrackbar{trackbarName[3], 'demo', alphaSliders[3], alphaSliderMax, onTrackbar}
cv.createTrackbar{trackbarName[4], 'demo', alphaSliders[4], alphaSliderMax, onTrackbar}

--Forward to model for segmentation
torch.setdefaulttensortype('torch.FloatTensor')

local maxLength = 640;

local modelPath = 'model.seg.base.multi-class.3f81535a2586a.19.t7'
if not paths.filep(modelPath) then
	error("Can't find file : " .. modelPath)
end

local Predictor = require('predictor')
local predictor = Predictor(modelPath, maxLength)

local function imgTrans(img, trans_type)
    -- trans_type = 1 (from H*W*C to C*H*W)
    -- trans_type = 2 (from C*H*W to H*W*C)
    
    ori_dim = img:size()
    ori_type = img:type()

    if trans_type == 1 then
       trans_img = torch.Tensor(ori_dim[3], ori_dim[1], ori_dim[2])
       trans_img = trans_img:type(ori_type)
       trans_img[{1,{},{}}] = img[{{},{},3}]
       trans_img[{2,{},{}}] = img[{{},{},2}]
       trans_img[{3,{},{}}] = img[{{},{},1}]
    else
       trans_img = torch.Tensor(ori_dim[2], ori_dim[3], ori_dim[1])
       trans_img = trans_img:type(ori_type)
       trans_img[{{},{},1}] = img[{3,{},{}}]
       trans_img[{{},{},2}] = img[{2,{},{}}]
       trans_img[{{},{},3}] = img[{1,{},{}}]
    end

    return trans_img
    
end

local imgsCPU = torch.FloatTensor()
while true do
    onTrackbar()
    for i=1,4 do
        curr_params[i] = capture:get{uvc_params[i]}
    end
    print('BRIGHTNESS:'..curr_params[1]..' CONTRAST:'..curr_params[2]..' SATURATION:'..curr_params[3]..' HUE:'..curr_params[4]..'\n')

    local _, frame = capture:read{}
    img = imgTrans(frame, 1)
    imgsCPU:resize(1, img:size(1), img:size(2), img:size(3))
    imgsCPU[1]:copy(img):div(255)
    img = predictor:predict(imgsCPU)[1]
    img = img:mul(255):round():byte()
    seg_frame = imgTrans(img, 2)
    cv.imshow{'demo', seg_frame}
    if cv.waitKey{30} >= 0 then break end
end

#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

// L22_background.cpp : 
//
// 运动目标检测 —— 背景建模 ///////////////////////
//
/////////////////////////////////////////////////
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 函数 refineSegments（） 来自 OpenCV 文档 
//static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
//{
//	int niters = 3;
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	Mat temp;
//	dilate(mask, temp, Mat(), Point(-1, -1), niters);
//	erode(temp, temp, Mat(), Point(-1, -1), niters * 2);
//	dilate(temp, temp, Mat(), Point(-1, -1), niters);
//	findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
//	dst = Mat::zeros(img.size(), CV_8UC3);
//	if (contours.size() == 0)
//		return;
//	// iterate through all the top-level contours,
//	// draw each connected component with its own random color
//	int idx = 0, largestComp = 0;
//	double maxArea = 0;
//	for (; idx >= 0; idx = hierarchy[idx][0])
//	{
//		const vector<Point>& c = contours[idx];
//		double area = fabs(contourArea(Mat(c)));
//		if (area > maxArea)
//		{
//			maxArea = area;
//			largestComp = idx;
//		}
//	}
//	Scalar color(0, 0, 255);
//	drawContours(dst, contours, largestComp, color, FILLED, LINE_8, hierarchy);
//}



/////////////////////////////////////////////////
//// 2. 背景建模 —— 平均背景法
//// （背景模型建立后不更新）
/////////////////////////////////////////////////

int main()
{
	VideoCapture cap;
	int number_to_train_on;
	int frame_count;
	int key, waittime;
	float a=0.5;
	Mat frame,buff,FrameAccSum, SqAccSum, avgFrame, avgDiff, IBackground;
	Mat IhiF, IlowF;
	Mat mask, masktmp;
	Mat tmp;

	// Float, 多个单通道矩阵
	vector<Mat> Igray(3);
	vector<Mat> Ilow(3);
	vector<Mat> Ihi(3);

	// 阈值
	float high_thresh;
	float low_thresh;

	///////////////////////////////////////////////////
	//  导入视频  ////////////////////////////////
	if (!cap.open("E://研究生教学资源//视觉信息处理与FPGA实现//作业//upload20210520//visiontraffic.avi"))  return EXIT_FAILURE;
	number_to_train_on = 50;
	waittime = 25;
	high_thresh = 40.0;
	low_thresh = 40.0;
	namedWindow("背景", WINDOW_AUTOSIZE);
	moveWindow("背景", 50, 401);
	namedWindow("视频", WINDOW_AUTOSIZE);
	moveWindow("视频", 50, 1);
	namedWindow("前景目标", WINDOW_AUTOSIZE);
	moveWindow("前景目标", 700, 1);

	//  背景建模  /////
	frame_count = 0;
	cout << "正在累积前" << number_to_train_on << "帧..." << endl;

	cap >> frame;

	Size sz = frame.size();
	FrameAccSum = Mat::zeros(sz, CV_32FC3);//对于累积值矩阵先初始化为0
	SqAccSum = Mat::zeros(sz, CV_32FC3);//对于平方和矩阵先初始化为0
	avgFrame = Mat::zeros(sz, CV_32FC3);
	avgDiff = Mat::zeros(sz, CV_32FC3);
	buff = Mat::zeros(sz, CV_32FC3);
	cout << "矩阵大小" << endl;//640*360
	cout << sz << endl;
	//cout << SqAccSum << endl;
	
	accumulate(frame, FrameAccSum);//做第一次累积
	accumulateSquare(frame, SqAccSum);//做第一次平方累积
	//cout << SqAccSum << endl; 
	cout << "第一次累积结束" << endl;
	frame_count++;
	avgFrame = FrameAccSum / frame_count;	//灰度平均值
	//cout << avgFrame << endl;
	avgDiff = (SqAccSum / frame_count) -
		(FrameAccSum.mul(FrameAccSum) / frame_count / frame_count);	//方差计算公式
	//cout << avgDiff << endl; 
	cout << "第一次计算结束" << endl;
	
	//cout << "第一次的累计值和平方值" << endl;
	//cout << FrameAccSum << endl;
	//cout << SqAccSum << endl;
	while (1)
	{//先读个50帧，计算参数

		cap >> frame;
		imshow("视频", frame);

		if (!frame.data)
		{
			cout << "视频结束" << endl;
			cout << "按任意键退出程序..." << endl;
			cap.release();
			waitKey(0);
			exit(0);
		}
		cout << "开始迭代" << endl;

		//FrameAccSum = 0;
		frame.convertTo(buff, CV_32FC3);
		//accumulate(frame, FrameAccSum);
		avgFrame = (1 - a)*avgFrame + a*buff; //均值迭代公式
		cout << "均值迭代结束" << endl;
		avgDiff = (1 - a)*avgDiff;
				+ a*((buff - avgFrame).mul(buff - avgFrame)); //方差迭代公式
		cout << "方差迭代结束" << endl;
		//FrameAccSum = 0;

		//accumulate(frame, FrameAccSum); 	//计算总灰度值
		//accumulateSquare(frame, SqAccSum);	//计算平方累加

		frame_count++;

		if ((key = waitKey(1)) == 27 || key == 'q' || key == 'Q' || frame_count >= number_to_train_on) break;
	}

	cout << "完成！" << endl;
	cout << "正在建立背景模型..." << endl;

	//cout << "累积后的灰度和方差平方" << endl;
	//cout << FrameAccSum << endl;
	//cout << SqAccSum << endl;

	//avgFrame = FrameAccSum / frame_count;	//灰度平均值
	//avgDiff = (SqAccSum / frame_count) -
		//(FrameAccSum.mul(FrameAccSum) / frame_count / frame_count);	//方差计算公式
	sqrt(avgDiff, avgDiff); //求方差开方
	//cout << "avgDiff开方值" << endl;
	//cout << avgDiff << endl;
	//cout << "计算后的灰度和方差" << endl;
	//cout << avgFrame << endl;
	//cout << avgDiff << endl;

	IhiF = avgFrame + (avgDiff + high_thresh); 	//得到阈值范围上限
	split(IhiF, Ihi);	//通道分离
	IlowF = avgFrame - (avgDiff + low_thresh);	//得到阈值范围下限
	split(IlowF, Ilow);	//通道分离

	avgFrame.convertTo(IBackground, CV_8UC3);
	imshow("背景", IBackground);

	cout << "完成背景建模! 按任意键开始检测前景" << endl;
	waitKey(0);
	///////////////////////////////////////////////////


	///////////////////////////////////////////////////
	//  前景检测  ////////////////////////////////
	// 按 esc 或 q 或 Q 退出检测
	cout << "检测前景中 ..." << endl;
	cout << "（esc 或 q 或 Q 退出检测）" << endl;

	//读入数据进行检测
	while ((key = waitKey(waittime)) != 27 || key == 'q' || key == 'Q')
	{
		cap >> frame;
		if (!frame.data) {
			cout << "检测结束！（esc 或 q 或 Q 退出检测）" << endl;
			cap.release();
			waitKey(0);
			exit(0);
		}

		frame.convertTo(tmp, CV_32FC3);
		split(tmp, Igray);
		// 判断各通道像素是否为背景
		// 通道 1
		inRange(Igray[0], Ilow[0], Ihi[0], mask);
		// 通道 2
		inRange(Igray[1], Ilow[1], Ihi[1], masktmp);
		mask = min(mask, masktmp);
		// 通道 3
		inRange(Igray[2], Ilow[2], Ihi[2], masktmp);
		mask = min(mask, masktmp);
		// 背景取反，标识前景
		//cout << mask << endl;
		mask = 255 - mask;

		// 显示视频图像 、 帧序号
		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		ss << cap.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		imshow("视频", frame);

		// 显示前景二值图像
		imshow("前景目标", mask);

	}
	cout << "按任意键退出程序" << endl;
	waitKey(0);
	cap.release();
	exit(0);
}


#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

// L22_background.cpp : 
//
// �˶�Ŀ���� ���� ������ģ ///////////////////////
//
/////////////////////////////////////////////////
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ���� refineSegments���� ���� OpenCV �ĵ� 
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
//// 2. ������ģ ���� ƽ��������
//// ������ģ�ͽ����󲻸��£�
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

	// Float, �����ͨ������
	vector<Mat> Igray(3);
	vector<Mat> Ilow(3);
	vector<Mat> Ihi(3);

	// ��ֵ
	float high_thresh;
	float low_thresh;

	///////////////////////////////////////////////////
	//  ������Ƶ  ////////////////////////////////
	if (!cap.open("E://�о�����ѧ��Դ//�Ӿ���Ϣ������FPGAʵ��//��ҵ//upload20210520//visiontraffic.avi"))  return EXIT_FAILURE;
	number_to_train_on = 50;
	waittime = 25;
	high_thresh = 40.0;
	low_thresh = 40.0;
	namedWindow("����", WINDOW_AUTOSIZE);
	moveWindow("����", 50, 401);
	namedWindow("��Ƶ", WINDOW_AUTOSIZE);
	moveWindow("��Ƶ", 50, 1);
	namedWindow("ǰ��Ŀ��", WINDOW_AUTOSIZE);
	moveWindow("ǰ��Ŀ��", 700, 1);

	//  ������ģ  /////
	frame_count = 0;
	cout << "�����ۻ�ǰ" << number_to_train_on << "֡..." << endl;

	cap >> frame;

	Size sz = frame.size();
	FrameAccSum = Mat::zeros(sz, CV_32FC3);//�����ۻ�ֵ�����ȳ�ʼ��Ϊ0
	SqAccSum = Mat::zeros(sz, CV_32FC3);//����ƽ���;����ȳ�ʼ��Ϊ0
	avgFrame = Mat::zeros(sz, CV_32FC3);
	avgDiff = Mat::zeros(sz, CV_32FC3);
	buff = Mat::zeros(sz, CV_32FC3);
	cout << "�����С" << endl;//640*360
	cout << sz << endl;
	//cout << SqAccSum << endl;
	
	accumulate(frame, FrameAccSum);//����һ���ۻ�
	accumulateSquare(frame, SqAccSum);//����һ��ƽ���ۻ�
	//cout << SqAccSum << endl; 
	cout << "��һ���ۻ�����" << endl;
	frame_count++;
	avgFrame = FrameAccSum / frame_count;	//�Ҷ�ƽ��ֵ
	//cout << avgFrame << endl;
	avgDiff = (SqAccSum / frame_count) -
		(FrameAccSum.mul(FrameAccSum) / frame_count / frame_count);	//������㹫ʽ
	//cout << avgDiff << endl; 
	cout << "��һ�μ������" << endl;
	
	//cout << "��һ�ε��ۼ�ֵ��ƽ��ֵ" << endl;
	//cout << FrameAccSum << endl;
	//cout << SqAccSum << endl;
	while (1)
	{//�ȶ���50֡���������

		cap >> frame;
		imshow("��Ƶ", frame);

		if (!frame.data)
		{
			cout << "��Ƶ����" << endl;
			cout << "��������˳�����..." << endl;
			cap.release();
			waitKey(0);
			exit(0);
		}
		cout << "��ʼ����" << endl;

		//FrameAccSum = 0;
		frame.convertTo(buff, CV_32FC3);
		//accumulate(frame, FrameAccSum);
		avgFrame = (1 - a)*avgFrame + a*buff; //��ֵ������ʽ
		cout << "��ֵ��������" << endl;
		avgDiff = (1 - a)*avgDiff;
				+ a*((buff - avgFrame).mul(buff - avgFrame)); //���������ʽ
		cout << "�����������" << endl;
		//FrameAccSum = 0;

		//accumulate(frame, FrameAccSum); 	//�����ܻҶ�ֵ
		//accumulateSquare(frame, SqAccSum);	//����ƽ���ۼ�

		frame_count++;

		if ((key = waitKey(1)) == 27 || key == 'q' || key == 'Q' || frame_count >= number_to_train_on) break;
	}

	cout << "��ɣ�" << endl;
	cout << "���ڽ�������ģ��..." << endl;

	//cout << "�ۻ���ĻҶȺͷ���ƽ��" << endl;
	//cout << FrameAccSum << endl;
	//cout << SqAccSum << endl;

	//avgFrame = FrameAccSum / frame_count;	//�Ҷ�ƽ��ֵ
	//avgDiff = (SqAccSum / frame_count) -
		//(FrameAccSum.mul(FrameAccSum) / frame_count / frame_count);	//������㹫ʽ
	sqrt(avgDiff, avgDiff); //�󷽲��
	//cout << "avgDiff����ֵ" << endl;
	//cout << avgDiff << endl;
	//cout << "�����ĻҶȺͷ���" << endl;
	//cout << avgFrame << endl;
	//cout << avgDiff << endl;

	IhiF = avgFrame + (avgDiff + high_thresh); 	//�õ���ֵ��Χ����
	split(IhiF, Ihi);	//ͨ������
	IlowF = avgFrame - (avgDiff + low_thresh);	//�õ���ֵ��Χ����
	split(IlowF, Ilow);	//ͨ������

	avgFrame.convertTo(IBackground, CV_8UC3);
	imshow("����", IBackground);

	cout << "��ɱ�����ģ! ���������ʼ���ǰ��" << endl;
	waitKey(0);
	///////////////////////////////////////////////////


	///////////////////////////////////////////////////
	//  ǰ�����  ////////////////////////////////
	// �� esc �� q �� Q �˳����
	cout << "���ǰ���� ..." << endl;
	cout << "��esc �� q �� Q �˳���⣩" << endl;

	//�������ݽ��м��
	while ((key = waitKey(waittime)) != 27 || key == 'q' || key == 'Q')
	{
		cap >> frame;
		if (!frame.data) {
			cout << "����������esc �� q �� Q �˳���⣩" << endl;
			cap.release();
			waitKey(0);
			exit(0);
		}

		frame.convertTo(tmp, CV_32FC3);
		split(tmp, Igray);
		// �жϸ�ͨ�������Ƿ�Ϊ����
		// ͨ�� 1
		inRange(Igray[0], Ilow[0], Ihi[0], mask);
		// ͨ�� 2
		inRange(Igray[1], Ilow[1], Ihi[1], masktmp);
		mask = min(mask, masktmp);
		// ͨ�� 3
		inRange(Igray[2], Ilow[2], Ihi[2], masktmp);
		mask = min(mask, masktmp);
		// ����ȡ������ʶǰ��
		//cout << mask << endl;
		mask = 255 - mask;

		// ��ʾ��Ƶͼ�� �� ֡���
		stringstream ss;
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		ss << cap.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		imshow("��Ƶ", frame);

		// ��ʾǰ����ֵͼ��
		imshow("ǰ��Ŀ��", mask);

	}
	cout << "��������˳�����" << endl;
	waitKey(0);
	cap.release();
	exit(0);
}


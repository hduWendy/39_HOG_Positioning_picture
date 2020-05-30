// 39_HOG_Positioning_picture.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
int main()
{
	//��ȡ���ص�����ͼƬ��������ͼƬ���ڵ�λ�ã��������
	cv::Mat Mat_0 = imread("C:/Users/lenovo/Desktop/��ͼͼƬ/img.png");
	cv::Mat tempMat = imread("C:/Users/lenovo/Desktop/��ͼͼƬ/template.png", 0);	
	cv::Mat Mat = imread("C:/Users/lenovo/Desktop/��ͼͼƬ/img.png", 0);
	cv::Mat Mat1;//Mat�е�һ����ͼ��

	tempMat.copyTo(Mat1);

	int height = tempMat.rows; //����
	int width = tempMat.cols; //ÿ��Ԫ�ص���Ԫ������
	int height_Mat = Mat.rows; //����
	int width_Mat = Mat.cols; //ÿ��Ԫ�ص���Ԫ������

	int cellSize = 16;
	int nX = height / cellSize;
	int nY = width / cellSize;
	int bins = nX * nY * 8;
	int cell_t = 0;

	int mX = height_Mat - height + 1;
	int mY = width_Mat - width + 1;
	int Mat_n_bins = mX * mY;

	cv::Mat gx, gy;
	cv::Mat mag, angle;

	//��ʼtempMat����
	//
	Sobel(tempMat, gx, CV_32F, 1, 0, 1);
	Sobel(tempMat, gy, CV_32F, 0, 1, 1);
	//�����ݶȺͽǶȷ���
	cartToPolar(gx, gy, mag, angle, true);
	//������̬����
	float* ref_hist = new float[bins];
	memset(ref_hist, 0, sizeof(float)*bins);
	//����Mat1���ƶȶ�̬����
	float* ref_hist_Mat1 = new float[bins];
	memset(ref_hist_Mat1, 0, sizeof(float)*bins);
	//��������Mat���ƶ�ƽ���̬����
	float* ref_hist_Mat_n = new float[Mat_n_bins];
	memset(ref_hist_Mat_n, 0, sizeof(float)*Mat_n_bins);

	for (int i = 0; i + cellSize <= height; i += cellSize)
	{
		for (int j = 0; j + cellSize <= width; j += cellSize)
		{
			//ÿһ��cell
			for (int det_i = 0; det_i < cellSize; det_i++)
			{
				for (int det_j = 0; det_j < cellSize; det_j++)
				{
					//ÿһ������
					for (int angle_t = 0; angle_t < 8; angle_t++)
					{
						if (angle.at<float>(i + det_i, j + det_j) >= angle_t * 45 && angle.at<float>(i + det_i, j + det_j) < (angle_t + 1) * 45)
						{
							ref_hist[cell_t + angle_t] += mag.at<float>(i + det_i, j + det_j);
							break;
						}
					}//�������ش������
				}
			}//����cell�������
			cell_t++;
		}
	}
	cell_t = 0;
	//tempMat�������

//	float L,M;

	//������ֵMat1
	int n = 0;
	int step_n = 1;//���������ģ�������Ҫ�޸�mX��mY
	int x, y;
	for (x = 0; x + height - 1 <= height_Mat - 1; x += step_n)
	{
		for (y = 0; y + width - 1 <= width_Mat - 1; y += step_n)
		{
			//��ֵÿһ��Mat1			
			for (int det_x = 0; det_x < height; det_x++)
			{
				for (int det_y = 0; det_y < width; det_y++)
				{
					//uint8_t sd = 0;
					//Mat1.at <uint8_t>(det_x, det_y) = sd;
					Mat1.at <uint8_t>(det_x, det_y) = Mat.at <uint8_t>(x + det_x, y + det_y);
				}
			}
			//��ʼMat1����
			//
			Sobel(Mat1, gx, CV_32F, 1, 0, 1);
			Sobel(Mat1, gy, CV_32F, 0, 1, 1);
			//�����ݶȺͽǶȷ���
			cartToPolar(gx, gy, mag, angle, true);

			for (int i = 0; i + cellSize <= height; i += cellSize)
			{
				for (int j = 0; j + cellSize <= width; j += cellSize)
				{
					//ÿһ��cell
					for (int det_i = 0; det_i < cellSize; det_i++)
					{
						for (int det_j = 0; det_j < cellSize; det_j++)
						{
							//ÿһ������
							for (int angle_t = 0; angle_t < 8; angle_t++)
							{
								if (angle.at<float>(i + det_i, j + det_j) >= angle_t * 45 && angle.at<float>(i + det_i, j + det_j) < (angle_t + 1) * 45)
								{
									ref_hist_Mat1[cell_t + angle_t] += mag.at<float>(i + det_i, j + det_j);
									break;
								}
							}//�������ش������
						}
					}//����cell�������
					cell_t++;
				}
			}
			cell_t = 0;
			//Mat1�������

			//����ŷ����þ��룬�õ�ֱ��ͼ�����ƶ�
			for (int t = 0; t < bins; t++)
			{
				ref_hist_Mat_n[n] += sqrt(pow((ref_hist_Mat1[t] - ref_hist[t]), 2));

//				M = sqrt(pow((ref_hist_Mat1[t] - ref_hist[t]), 2));

			}
//			L = ref_hist_Mat_n[n];	

			//����һ��Mat1
			n++;
			//����Mat1���ƶȶ�̬����
			memset(ref_hist_Mat1, 0, sizeof(float)*bins);
		}
	}

	//Ѱ�����ƶ����ֵ
	float Min_ref_hist_Mat_n = ref_hist_Mat_n[0];
	int Min_n = 0;
	for (int n = 20000; n < Mat_n_bins; n++)
	{
//		float N = ref_hist_Mat_n[n];
		if (ref_hist_Mat_n[n] < Min_ref_hist_Mat_n)
		{
			Min_ref_hist_Mat_n = ref_hist_Mat_n[n];
			Min_n = n;
		}
	}

	//���������Ͻ�����
	int rect_x = 0;int rect_y = 0;
	if ((Min_n + 1) % mY == 0)
	{
		rect_x = ((Min_n + 1) / mY - 1) * step_n - 1;
		rect_y = (mY - 1) * step_n;		
	}
	else
	{
		rect_x = (Min_n + 1) / mY * step_n;
		rect_y = ((Min_n + 1) % mY - 1) * step_n;
	}

	//����
	cv::Rect rect;
	rect.x = rect_y;
	rect.y = rect_x;
	rect.width = width;
	rect.height = height;
	rectangle(Mat_0, rect, CV_RGB(0, 255, 0), 1, 8, 0);

	//��ʾ
	imshow("Mat", Mat_0); 

	//�ȴ��û�����
	waitKey(0);
	delete[] ref_hist;
	delete[] ref_hist_Mat1;
	delete[] ref_hist_Mat_n;
	return 0;
}


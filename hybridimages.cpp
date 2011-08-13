#include <stdio.h>
#include <math.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

/////////////////////////
////  パラメータ類   
/////////////////////////
#define SIGMA 7.5            // ガウシアンカーネルの標準偏差パラメータ．これが大きいとINPUT1がより強く見える
#define SIDE_MAX 400        // INPUT1のどちらか1辺のサイズがこれより大きければリサイズ

void swap(cv::Mat& src, const double xm, const double ym) {
    const int cx = static_cast<int>(xm), cy = static_cast<int>(ym);
    cv::Mat q1, q2, q3, q4, tmp(src.rows/2, src.cols/2, CV_32FC2);
    q1 = src(cv::Rect( 0,  0, cx, cy));
    q2 = src(cv::Rect(cx,  0, cx, cy));
    q3 = src(cv::Rect(cx, cy, cx, cy));
    q4 = src(cv::Rect( 0, cy, cx, cy));
    q3.copyTo(tmp);
    q1.copyTo(q3);
    tmp.copyTo(q1);
    q4.copyTo(tmp);
    q2.copyTo(q4);
    tmp.copyTo(q2);
}

////////////////////////////////////
//// ガウシアンカーネル作成
////////////////////////////////////
void myGauss(CvMat* gau1, CvMat* gau2, const float sig){
    cv::Mat g1(gau1), g2(gau2);
    cv::Size ksize;
    ksize.width = gau1->width;
    ksize.height = gau1->height;
    
    const float ss2 = sig*sig*2;
    const int xs = ksize.width, ys = ksize.height;
    const float xm = xs/2.0, ym = ys/2.0;
    for (int y = 0; y < ys; y++) {
        float* row1 = g1.ptr<float>(y);
        float* row2 = g2.ptr<float>(y);
        for (int x = 0; x < xs; x++) {
            double gaud = (xm-x)*(xm-x)+(ym-y)*(ym-y);
            double gaui = exp(-gaud/ss2);
            row1[x*2] = gaui;
            row1[x*2+1] = gaui;
            row2[x*2] = 1-gaui;
            row2[x*2+1] = 1-gaui;
        }
    }
    swap(g1, xm, ym);
    swap(g2, xm, ym);    
    
    *gau1 = (CvMat)g1;
    *gau2 = (CvMat)g2;
}


//////////////////////////////////////////
//// 1チャンネルでの Hybrid images 生成
//////////////////////////////////////////
void HybridImages1ch(
	CvMat* src1,
	CvMat* src2, 
	CvMat* dft_A,
	CvMat* dft_B, 
	const CvMat* gau1, 
	const CvMat* gau2, 
	const CvMat* srcIm )
{
	// 入力画像と虚数配列をマージして複素数平面を構成
	CvMat* complex1 = cvCreateMat( src1->rows, src1->cols, CV_32FC2 ); 
	CvMat* complex2 = cvCreateMat( src1->rows, src1->cols, CV_32FC2 );
	cvMerge(src1, srcIm, NULL, NULL,complex1);
	cvMerge(src2, srcIm, NULL, NULL,complex2);

	CvMat tmp;
	// 複素数平面をdft_A, dft_Bにコピーし，残りの行列右側部分を0で埋めた後，離散フーリエ変換を行う
	cvGetSubRect( dft_A, &tmp, cvRect(0,0,complex1->cols,complex1->rows));
	cvCopy( complex1, &tmp );
	if(dft_A->cols > complex1->cols){
		cvGetSubRect( dft_A, &tmp, cvRect(complex1->cols,0,dft_A->cols - complex1->cols,complex1->rows));
		cvZero( &tmp );
	}
	cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complex1->rows );

	
	cvGetSubRect( dft_B, &tmp, cvRect(0,0,complex1->cols,complex1->rows));
	cvCopy( complex2, &tmp );
	if(dft_A->cols > complex1->cols){
		cvGetSubRect( dft_B, &tmp, cvRect(complex1->cols,0,dft_B->cols - complex1->cols,complex1->rows));
		cvZero( &tmp );
	}
	cvDFT( dft_B, dft_B, CV_DXT_FORWARD, complex1->rows );

	// 周波数領域において，ガウシアンカーネルGを用いて以下の処理を行う
	// Output = Input1・G + Input2・(1-G)
	cvMul(dft_A,gau1,dft_A);
	cvMul(dft_B,gau2,dft_B);
	cvAdd(dft_A,dft_B,dft_A);

	// 離散フーリエ逆変換し，実数成分をsrc1に，虚数成分をsrc2に格納
	cvDFT( dft_A, dft_A, CV_DXT_INV_SCALE, src1->rows ); // 上部のみを計算する
	cvGetSubRect( dft_A, &tmp, cvRect(0,0,src1->cols,src1->rows) );
	cvCopy( &tmp, complex1 );
	cvSplit(complex1,src1,src2,0,0);
	cvReleaseMat(&complex1);
	cvReleaseMat(&complex2);
}


//////////////////////////////
////  Hybrid images 生成
//////////////////////////////
void HybridImages(CvMat* src1, const CvMat* src2, const float sig) {
    //行列確保
    int dft_M = cvGetOptimalDFTSize( src1->height ); 
    int dft_N = cvGetOptimalDFTSize( src1->width  ); 
    CvMat* src1b = cvCreateMat( src1->rows, src1->cols, CV_32FC1 ); 
    CvMat* src1g = cvCreateMat( src1->rows, src1->cols, CV_32FC1 ); 
    CvMat* src1r = cvCreateMat( src1->rows, src1->cols, CV_32FC1 ); 
    CvMat* src2b = cvCreateMat(src2->rows, src2->cols, CV_32FC1 ); 
    CvMat* src2g = cvCreateMat(src2->rows, src2->cols, CV_32FC1 ); 
    CvMat* src2r = cvCreateMat(src2->rows, src2->cols, CV_32FC1 ); 
    CvMat* srcIm = cvCreateMat(src2->rows, src2->cols, CV_32FC1 ); 
    CvMat* dft_A = cvCreateMat( dft_M, dft_N, CV_32FC2 ); 
    CvMat* dft_B = cvCreateMat( dft_M, dft_N, CV_32FC2 );
    CvMat* gau1 = cvCreateMat( dft_M, dft_N, CV_32FC2 );
    CvMat* gau2 = cvCreateMat( dft_M, dft_N, CV_32FC2 );
    myGauss(gau1,gau2,sig);
    cvZero(srcIm);
    
    //各チャンネルごとに処理
    cvSplit(src1,src1b,src1g,src1r,0);
    cvSplit(src2,src2b,src2g,src2r,0);
    HybridImages1ch(src1b, src2b, dft_A, dft_B, gau1, gau2,srcIm);
    HybridImages1ch(src1g, src2g, dft_A, dft_B, gau1, gau2,srcIm);
    HybridImages1ch(src1r, src2r, dft_A, dft_B, gau1, gau2,srcIm);
    cvMerge(src1b, src1g, src1r, NULL,src1);
    
    //メモリの解放
    cvReleaseMat(&src1b);
    cvReleaseMat(&src1g);
    cvReleaseMat(&src1r);
    cvReleaseMat(&src2b);
    cvReleaseMat(&src2g);
    cvReleaseMat(&src2r);
    cvReleaseMat(&srcIm);
    cvReleaseMat(&dft_A);
    cvReleaseMat(&dft_B);
    cvReleaseMat(&gau1);
    cvReleaseMat(&gau2);
}

/////////////////////////////////////////
//// 画像を読み込んでHybrid imagesを保存
/////////////////////////////////////////
int HybridImages_main(const std::string& fname1, const std::string& fname2, const std::string& outname, float sig) {
    IplImage *img1,*img2,*result,*dst1,*dst2, *res,img_hdr;
    int width,height;
    CvMat istub1, *src1,istub2,*src2;
    
    //INPUT1
    if((img1=cvLoadImage(fname1.c_str(), CV_LOAD_IMAGE_COLOR ))==0){
        return -1;
    }
    if(img1->width > img1->height){
        if(img1->width > SIDE_MAX){
            width=SIDE_MAX;
            height = SIDE_MAX* img1->height / img1->width;
        }else{
            width =img1->width;
            height = img1->height;		
        }	
    }else{
        if(img1->height > SIDE_MAX){
            height=SIDE_MAX;
            width = SIDE_MAX* img1->width / img1->height;
        }else{
            width =img1->width;
            height = img1->height;		
        }
    }
    dst1 =cvCreateImage(cvSize(width,height),IPL_DEPTH_8U, 3);
    cvResize(img1,dst1,CV_INTER_CUBIC);
    cvReleaseImage(&img1);
    img1 = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 3);
    cvConvertScale(dst1, img1);
    src1 = cvGetMat(img1, &istub1);
    cvReleaseImage(&dst1);
    
    //INPUT2
    if( (img2 = cvLoadImage(fname2.c_str(), CV_LOAD_IMAGE_COLOR ) )==0){
        return -1;
    }
    dst2 =cvCreateImage(cvSize(width,height),IPL_DEPTH_8U, 3);
    cvResize(img2,dst2,CV_INTER_CUBIC); 
    cvReleaseImage(&img2);
    img2 = cvCreateImage(cvSize(width,height), IPL_DEPTH_32F, 3);
    cvConvertScale(dst2, img2);
    src2 = cvGetMat(img2, &istub2);
    cvReleaseImage(&dst2);

    //Hybrid images 生成処理を呼び出し，結果画像を保存
    HybridImages(src1, src2, sig);
    result = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 3);
    res = cvGetImage(src1, &img_hdr);
    cvConvertScaleAbs(res, result);
    if( (cvSaveImage(outname.c_str(), result))==0){
        return -1;
    }
    return 1;
}


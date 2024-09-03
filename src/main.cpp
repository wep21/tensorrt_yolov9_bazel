#include "opencv2/opencv.hpp"

int main(int argh, char* argv[]) {
  cv::VideoCapture cap(0);

  if (!cap.isOpened()) {
    return -1;
  }

  cv::Mat frame;
  while (cap.read(frame)) {
    cv::imshow("win", frame);
    const int key = cv::waitKey(1);
    if (key == 'q') {
      break;
    } else if (key == 's') {
      cv::imwrite("img.png", frame);
    }
  }
  cv::destroyAllWindows();
  return 0;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

std::string detectShape(const std::vector<cv::Point>& approx) {
    int vertices = approx.size();
    if (vertices == 3) return "Triangle";
    else if (vertices == 4) {
        cv::Rect box = cv::boundingRect(approx);
        double aspectRatio = static_cast<double>(box.width) / box.height;
        return (aspectRatio > 0.95 && aspectRatio < 1.05) ? "Square" : "Rectangle";
    } else if (vertices > 6) {
        return "Circle";
    }
    return "Polygon";
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the webcam\n";
        return -1;
    }

    cv::Mat frame, gray, blurred, edges;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approx;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 2.0);
        cv::Canny(blurred, edges, 50, 150);

        contours.clear();
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 100) continue;

            cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);

            if (cv::isContourConvex(approx)) {
                std::string shape = detectShape(approx);

                cv::drawContours(frame, std::vector<std::vector<cv::Point>>{approx}, -1, cv::Scalar(0, 255, 0), 2);

                cv::Moments M = cv::moments(approx);
                if (M.m00 != 0) {
                    int cx = static_cast<int>(M.m10 / M.m00);
                    int cy = static_cast<int>(M.m01 / M.m00);
                    cv::putText(frame, shape, cv::Point(cx - 30, cy), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        cv::imshow("Shape Detection", frame);
        cv::imshow("Edges", edges);

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

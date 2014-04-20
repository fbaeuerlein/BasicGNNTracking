#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "KalmanFilter.h"
#include "Tracker.h"
using namespace std;
using namespace cv;

bool mouseDown = false;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if ( event == CV_EVENT_LBUTTONDOWN )  { mouseDown = true; }
     if ( event == CV_EVENT_LBUTTONUP ) { mouseDown = false; }

     if ( event == EVENT_MOUSEMOVE && mouseDown )
     {
        *((int*) userdata) = x;
        *( ( (int*) userdata ) + 1) = y;
     }

     if ( !mouseDown )
     {
         *((int*) userdata) = -1;
         *( ( (int*) userdata ) + 1) = -1;
     }
}

int main()
{
    std::cout << "###############################" << std::endl;
    std::cout << "press q to quit" << std::endl;
    std::cout << "left click and hold mouse on image to create additional measurement" << std::endl;

    // 4-dimensional state, 2-dimensional measurements
    typedef Tracker<4, 2> GNNTracker;
    GNNTracker tracker;

    typedef GNNTracker::MeasurementSpaceVector Measurement;
    typedef GNNTracker::Measurements Measurements;

    namedWindow( "Kalman Demo", CV_WINDOW_AUTOSIZE );

    int * mouse = new int[2]();
    mouse[0] = mouse[1] = -1;

    setMouseCallback("Kalman Demo",CallBackFunc, mouse);

    int k = 0;
    size_t step = 0;

    while ( k != 1048689 ) //'q'
    {
        Mat img(480, 640, CV_8UC3, Scalar(255, 255, 255));

        Measurements m;

        // build some virtual measurements
        m.emplace_back(Measurement(320 + 120 * std::sin(step * .05), 240 + 100 * std::cos(step * .05)));
        m.emplace_back(Measurement(240 - 120 * std::sin(step * .05), 240 + 100 * std::cos(step * .05)));
        m.emplace_back(Measurement(320 - 120 * std::sin(step * .05), 240 - 200 * std::cos(step * .05)));

        // get measurement from mouse coordinates
        if (mouse[0] != -1 && mouse[1] != -1)
            m.emplace_back(Measurement(mouse[0], mouse[1]));

        step++;

        for ( const auto & x : m )
        {
            const int r = 5;
            // draw measurement
            circle(img, Point(x(0), x(1)), 2*r, Scalar(0, 0, 0));
            line(img, Point(x(0) - r, x(1) - r), Point(x(0) + r, x(1) + r), Scalar(0, 0, 0) );
            line(img, Point(x(0) - r, x(1) + r), Point(x(0) + r, x(1) - r), Scalar(0, 0, 0) );
        }

        tracker.track(m);

        for ( const auto & filter : tracker.filters() )
        {

            const GNNTracker::StateSpaceVector s = filter.state();
            const GNNTracker::StateSpaceVector p = filter.prediction();

            // draw filter position
            circle(img, Point(s(0), s(1)), 5, Scalar(0, 0, 255));

            // draw filter prediction
            circle(img, Point(p(0), p(1)), 5, Scalar(255, 0, 0));

            // draw filter velocity (scaled)
            line(img, Point(s(0), s(1)), Point(s(0) + 5 * s(2), s(1) + 5 * s(3)), Scalar(0, 255, 0));

            stringstream id;

            id << filter.id();

            putText(img, id.str(), cv::Point2i(s(0) + 10, s(1)), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));

        }

        imshow( "Kalman Demo", img );

        k = cv::waitKey(30);
    }
    return 0;
}


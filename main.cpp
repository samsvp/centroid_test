#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>

#include "tracker.hpp"

std::vector<cv::Point2f> calculate_centers(cv::Mat src, 
    std::vector<cv::Vec4i>& hierarchy,
    std::vector<std::vector<cv::Point>>& contours,
    bool show_bw=0)
{
   
    // Create binary image from source image
    cv::Mat bw;
    cv::cvtColor(src, bw, cv::COLOR_BGR2GRAY);
    cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    if (show_bw) cv::imshow("Binary Image", bw);
    
    
    // find contours
    cv::findContours(bw, contours, hierarchy, 
        cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0) 
    );

    // remove small regions of the image
    contours.erase(
        std::remove_if(contours.begin(), contours.end(),
            [](std::vector<cv::Point> p){ return cv::contourArea(p) < 100; }),
        contours.end()
    );

    // get the moments
    std::vector<cv::Moments> mu(contours.size());
    for(int i = 0; i<contours.size(); i++)
    { 
        mu[i] = cv::moments( contours[i], false ); 
    }

    // get the centroid of figures.
    std::vector<cv::Point2f> mc(contours.size());
    for(int i = 0; i<contours.size(); i++)
    {
        mc[i] = cv::Point2f(
            mu[i].m10/(mu[i].m00+0.0001), 
            mu[i].m01/(mu[i].m00+0.0001)
        ); 
    }

    return mc;
}


std::vector<cv::Point2f> calculate_centers(cv::Mat src)
{
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    return calculate_centers(src, hierarchy, contours);
}


std::vector<float> calc_dists(
    cv::Point2f p, const std::vector<cv::Point2f>& points)
{
    std::vector<float> distances(points.size());
    
    std::transform(points.begin(), points.end(), distances.begin(),
        [&p](cv::Point2f _p){ return abs(p.x - _p.x) + abs(p.y- _p.y); }
    );

    return distances;
}


int find_min(int i, std::vector<cv::Point2f> new_points,
        std::vector<cv::Point2f> old_points)
{
    if (new_points.empty() || old_points.empty()) return -1;

    cv::Point2f p = new_points[i];
    // distance from this point to the old points
    std::vector<float> pdists = calc_dists(p, old_points);
    int pargmin = std::min_element(pdists.begin(), pdists.end()) - pdists.begin();
    // distance from target point to other points
    cv::Point2f target_p = old_points[pargmin];
    std::vector<float> odists = calc_dists(target_p, new_points);
    int oargmin = std::min_element(odists.begin(), odists.end()) - odists.begin();
    // found new point position
    if (oargmin == i)
    {
        return pargmin;
    }
    // did not found new point position; remove old points
    else
    {
        new_points.erase(new_points.begin() + oargmin);
        old_points.erase(old_points.begin() + pargmin);
        // update index
        int new_i = std::find(new_points.begin(), new_points.end(), p) - new_points.begin();
        return find_min(new_i, new_points, old_points);
    }
}


int main(int argc, char *argv[])
{

    Tracker tracker;
    // cv::Point2f p1(1,1);
    // cv::Point2f p2(4,1);
    // cv::Point2f p3(5,5);
    // cv::Point2f p4(2,2);

    // std::vector<cv::Point2f> old_points = {p1, p2, p3};
    // std::vector<cv::Point2f> new_points = {p2, p1, p4, p3};

    // for (int i=0; i<new_points.size(); i++)
    // {
    //     auto p = new_points[i];
    //     int index = tracker.find_min(i, new_points, old_points);
    //     std::cout << index << "\n";
    // }

    // Load the image
    cv::CommandLineParser parser( argc, argv, "{@input | cards.png | input image}" );
    cv::Mat src = imread( parser.get<cv::String>( "@input" ) );
    if(src.empty())
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
        std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
        return -1;
    }
    

    tracker.update(src);
    auto contours = tracker.get_countours();
    auto mc = tracker.get_centers();
    auto hierarchy = tracker.get_hierarchy();

    // draw contours
    cv::Mat drawing(src.size(), CV_8UC3, cv::Scalar(255,255,255));
    for(int i = 0; i<contours.size(); i++)
    {
        auto color = tracker.get_color(i);
        cv::drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
        cv::circle( drawing, mc[i], 4, color, -1, 8, 0 );
    }
    
    // show the resultant image
    cv::namedWindow("Contours", CV_WINDOW_NORMAL);
    cv::imshow("Contours", drawing);
    cv::waitKey(0);

    return 0;
}
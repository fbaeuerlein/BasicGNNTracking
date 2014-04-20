BasicGNNTracking
================

This shows a basic implementation of the global nearest neighbour (GNN) multi target Tracker. Kalman filter is used for Tracking and Auction Algorithm for determining the assignment of measurments to filters.

Generates some sample measurements of virtual targets that move on elliptical trajectories. 
You can create further measurements with your mouse by holding the left mouse button and moving the pointer around.

Basic functionality: 
creates filters for unassigned measurements and assigns measurements to existing filters by maximizing the sum of probabilities of
assignments (i.e. minimizing the mahalanobis-distance) with the auction algorithm. Unassigned filters will be discarded.

Change parameters for covariances and probabilities in Tracker.h to fit your needs ...

Further information on GNN-trackers could be found at http://ecet.ecs.uni-ruse.bg/cst/docs/proceedings/S3/III-7.pdf

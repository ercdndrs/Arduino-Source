/*  Overworld Target Tracker
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#ifndef PokemonAutomation_PokemonSwSh_OverworldTargetTracker_H
#define PokemonAutomation_PokemonSwSh_OverworldTargetTracker_H

#include <deque>
#include "Common/Clientside/SpinLock.h"
#include "CommonFramework/Tools/Logger.h"
#include "CommonFramework/Tools/VideoFeed.h"
#include "CommonFramework/Inference/VisualInferenceCallback.h"
#include "PokemonSwSh/Programs/PokemonSwSh_OverworldTrajectory.h"

namespace PokemonAutomation{
namespace NintendoSwitch{
namespace PokemonSwSh{



enum class OverworldMark{
    EXCLAMATION_MARK,
    QUESTION_MARK,
};

struct OverworldTarget{
    OverworldMark mark;
    InferenceBox box;
    Trajectory trajectory;
    double delta_x;
    double delta_y;
};


class OverworldTargetTracker : public VisualInferenceCallbackWithCommandStop{
public:
    static const double OVERWORLD_CENTER_X;
    static const double OVERWORLD_CENTER_Y;

public:
    OverworldTargetTracker(
        Logger& logger, VideoFeed& feed,
        std::chrono::milliseconds window,
        double mark_offset,
        bool prioritize_exclamations,
        double max_alpha
    );

    //  If set to true, this inference object will not return true on
    //  "on_frame()" callbacks.
    void set_stop_on_target(bool stop);

    void clear_detections();

    //  Get the best target as of right now.
    //  The return value is only valid if the first element is non-negative.
    std::pair<double, OverworldTarget> best_target();

    virtual bool on_frame(
        const QImage& frame,
        std::chrono::system_clock::time_point timestamp
    ) override final;


private:
    struct Mark{
        std::chrono::system_clock::time_point timestamp;
        InferenceBox box;
    };

    static void populate_targets(
        std::multimap<double, OverworldTarget>& scored_targets,
        const std::vector<OverworldTarget>& targets
    );
    void populate_targets(
        std::multimap<double, OverworldTarget>& scored_targets,
        const std::deque<Mark>& marks,
        OverworldMark mark
    );

    bool save_target(std::multimap<double, OverworldTarget>::iterator target);


private:
    Logger& m_logger;
    VideoFeed& m_feed;
    std::chrono::milliseconds m_window;
    double m_mark_offset;
    bool m_prioritize_exclamations;
    double m_max_alpha;

    InferenceBoxScope m_search_area;
    std::deque<InferenceBoxScope> m_detection_boxes;

    //  Sliding window of detections.
    std::deque<Mark> m_exclamations;
    std::deque<Mark> m_questions;

    std::atomic<bool> m_stop_on_target;
    SpinLock m_lock;
    std::pair<double, OverworldTarget> m_best_target;
};





}
}
}
#endif

/*  Program Environment
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#include "ProgramEnvironment.h"


namespace PokemonAutomation{


ProgramEnvironment::ProgramEnvironment(
    Logger& logger,
    StatsTracker* current_stats,
    const StatsTracker* historical_stats
)
    : m_enable_feedback(true)
    , m_stopping(false)
    , m_logger(logger)
    , m_current_stats(current_stats)
    , m_historical_stats(historical_stats)
{}



void ProgramEnvironment::update_stats(){
    std::string current;
    if (m_current_stats){
        current = m_current_stats->to_str();
    }
    std::string historical;
    if (m_historical_stats){
        historical = m_historical_stats->to_str();
    }

    if (current.empty() && historical.empty()){
        set_status("");
        return;
    }

    if (!current.empty() && historical.empty()){
        QString str = QString(current.c_str());
        set_status(str);
        log(str);
        return;
    }
    if (current.empty() && !historical.empty()){
        QString str = QString(historical.c_str());
        set_status("<b>Past Runs</b> - " + str);
        return;
    }

    log(QString(current.c_str()));

    std::string str;
    str += "<b>Current Run</b> - " + current;
    str += "<br>";
    str += "<b>Past Totals</b> - " + historical;

    set_status(str.c_str());
}

bool ProgramEnvironment::is_stopping() const{
    return m_stopping.load(std::memory_order_acquire);
}
void ProgramEnvironment::check_stopping() const{
    if (is_stopping()){
        throw PokemonAutomation::CancelledException();
    }
}
void ProgramEnvironment::signal_stop(){
    m_stopping.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lg(m_lock);
    m_cv.notify_all();
}




}

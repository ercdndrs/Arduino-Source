/*  UI List for all the Programs
 *
 *  From: https://github.com/PokemonAutomation/Arduino-Source
 *
 */

#ifndef PokemonAutomation_ProgramListUI_H
#define PokemonAutomation_ProgramListUI_H

#include <QListWidget>
#include <QVBoxLayout>

namespace PokemonAutomation{


class MainWindow;

class ProgramListUI : public QListWidget{
    Q_OBJECT

public:
    ProgramListUI(MainWindow& parent);
    int text_width() const{ return m_text_width; }

public slots:
    void row_selected(QListWidgetItem* item);
    void row_changed(int row);

private:
    MainWindow& m_parent;
    int m_text_width;
};


}
#endif

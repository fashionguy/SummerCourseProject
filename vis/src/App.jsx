import React from "react"
import * as d3 from "d3"
import "./App.css"
import { Col, Row } from "antd"
import Snapshots from "./snapshots"
import Graph from "./graph"

class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            snapshots: [],
            graph: {},
            index: 2017

        };
        // this.changeIndex = this.changeIndex.bind(this);
        // this.inputChange = this.inputChange.bind(this);
    }

    componentDidMount() {
        d3.json("./tsne_data.json").then(snapshots => {
            this.setState({
                snapshots: snapshots,
                graph: snapshots[this.state.index].graph
            })
        });
    }
 
    changeIndex(i){
        
        this.setState({
            snapshots: this.state.snapshots,
            graph: this.state.snapshots[i].graph,
            index: i
        });
        
    }

    increaseIndex(){
        
        if(this.state.index < 2017){
            this.setState({
                graph: this.state.snapshots[this.state.index + 1].graph,
                index: this.state.index + 1
            });
        }else{
            alert("找bug呢？你不能下一个了")
        }        
    }

    decreaseIndex(){
        if(this.state.index > 0){
            this.setState({
                graph: this.state.snapshots[this.state.index - 1].graph,
                index: this.state.index - 1
            });
        }else{
            alert("找bug呢？你不能在上一个了")
        }
        
        
    }

    inputChange(event){
    
        // this.setState({
        //     graph: this.state.snapshots[parseInt(event.target.value)].graph,
        //     index: parseInt(event.target.value)
        // });
        console.log(event.target.value)
    }

    componentWillUnmount() {
        clearInterval(this.id);
    }
 
    
    render() {
        const snapshots = this.state.snapshots
        const graph = this.state.graph
        return (
            <div className="App">

                <h2>左边为2018个时间窗，右边为所选时间窗对应的联系图。</h2>
                <h2>现在是第{this.state.index}个时间窗。</h2>
                
                <input type="button" value="上一个时间窗" onClick={ () => this.decreaseIndex()} />
                <input type="button" value="下一个时间窗" onClick={ () => this.increaseIndex()} />

                <Row>
                    <Col span={12}>
                        <Snapshots snapshots={snapshots} changeIndex={ i => this.changeIndex(i) } />
                    </Col>
                    <Col span={12}>
                        <Graph graph={graph} />
                    </Col>
                </Row>
            </div>
                
        )
    }
    
}

export default App
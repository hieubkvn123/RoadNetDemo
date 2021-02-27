/* Main interface */
import React, { Component } from "react"
import axios from 'axios'

/* Bootstrap components */
import {
	Modal, Button
} from 'react-bootstrap'

/* All local dependencies */
import config from "../config"
import countries from './countries'

/* Bootstrap + CSS */
import "./css/main.css"
import 'bootstrap/dist/css/bootstrap.min.css';

/* Import all OpenLayer dependencies */
import "ol/ol.css"
import BingMaps from "ol/source/BingMaps"
import Map from "ol/Map"
import TileLayer from "ol/layer/Tile"
import View from "ol/View"
import {fromLonLat, toLonLat} from "ol/proj"

class MapView extends Component {
	constructor(props){
		// inside props we store 
		// - _styles : map styles for display
		// - _layers : layers for each style
		// - _view   : the view for each layer
		// - map     : the canvas to draw the map
		super(props)
		this.countries = countries

		// define the states, store the following info :
		// - _long : the current center longtitude of the map
		// - _lat  : the current center latitude of the map
		this.state = {
			_long : config['default_long'],
			_lat  : config['default_lat'],
			_search_long : config['default_long'],
			_search_lat : config['default_lat'],
			_show_model_dialog : false,
			_show_map_dialog : false,
			models : [],	
			current_map_url : ''
		}

		this.render = this.render.bind(this)
		this.componentDidMount = this.componentDidMount.bind(this)

		/* Event handlers */
		this.onChange = this.onChange.bind(this)
		this.onCountryChange = this.onCountryChange.bind(this)
		this.onCoordChange = this.onCoordChange.bind(this)
		this.onSearchCoords = this.onSearchCoords.bind(this)
		this.onLongLatChange = this.onLongLatChange.bind(this)
		this.onDefaultCoords = this.onDefaultCoords.bind(this)
		this.handleHideMapDialog = this.handleHideMapDialog.bind(this)

		/* Util functions */
		this.showModelsDialog = this.showModelsDialog.bind(this)
		this.setModelsList = this.setModelsList.bind(this)
		this.onModelSelected = this.onModelSelected.bind(this)
		this.exportToImage = this.exportToImage.bind(this)
		this.getImageURL = this.getImageURL.bind(this)
		this.sendCanvasToServer = this.sendCanvasToServer.bind(this)
	}

	componentDidMount(){
		// Config all layers of OpenLayer
		this._styles = [
		  'Aerial',
		  'AerialWithLabelsOnDemand',
		  'RoadOnDemand',
		  'CanvasDark',
		];
		
		this._layers = [];
		var _i, _ii;
		for (_i = 0, _ii = this._styles.length; _i < _ii; ++_i) {
		  this._layers.push(
			new TileLayer({
			  visible: false,
			  preload: Infinity,
			  source: new BingMaps({
				key: config['bing_map_api_key'],
				imagerySet: this._styles[_i],
				// use maxZoom 19 to see stretched tiles instead of the BingMaps
				// "no photos at this zoom level" tiles
				// maxZoom: 19
			  }),
			})
		  );
		}

		// long-lat coordinates
		var washingtonLonLat = [this.state._long, this.state._lat] // Singapore default coordinate
		var washingtonWebMercator = fromLonLat(washingtonLonLat);

		this._view = new View({
			projection : config['map_projection_system'],
			center : washingtonWebMercator,
			zoom : 13
		})

		// Make _long & _lat update on change 
		this._view.on('change:center', this.onCoordChange);

		this.map = new Map({
			layers : this._layers,
			target : 'map',
			view : this._view
		})

		// Make AerialWithLabelsOnDemand visible by default
		this._layers[1].setVisible(true)

		// Get existing models
		axios({
			url : `http://${config['compute_server_ip']}:${config['compute_server_port']}/get_models_list`,
			method : 'POST',
			headers : {
				'Content-Type' : 'multipart/form-data'
			}
		}).then(response => response.data)
		.then(response => this.setModelsList(response))
		.catch(err => console.log(err))
	}


	/* --Event handlers-- */
	onCoordChange() {
		//var current_coord = [this._view.center[0], this._view.center[1]]
		var current_long_lat = toLonLat(this._view.getCenter())

		// console.log(this._view.center)
		this.setState({_long : current_long_lat[0].toFixed(4)})
		this.setState({_lat  : current_long_lat[1].toFixed(4)})
	}

	onChange(e) {
		var style = e.target.value;
		for (var i = 0, ii = this._layers.length; i < ii; ++i) {
			this._layers[i].setVisible(this._styles[i] === style);
		}
	}

	onCountryChange(e){
		var country_id = e.target.value
		var country_lon = this.countries[country_id].longitude
		var country_lat = this.countries[country_id].latitude

		this.setState({_search_long : country_lon})
		this.setState({_search_lat : country_lat})
	}

	onSearchCoords(e) {
		 var _search_long = this.state._search_long
		 var _search_lat  = this.state._search_lat

		 var center_lonlat = [_search_long, _search_lat]
		 var coords = fromLonLat(center_lonlat)

		 this._view.setCenter(coords)
	}

	onDefaultCoords() {
		var center_lonlat = [config['default_long'], config['default_lat']]
		var coords = fromLonLat(center_lonlat)

		this._view.setCenter(coords)
	}

	onLongLatChange(e) {
		this.setState({[e.target.name] : e.target.value})
	}

	sendCanvasToServer(canvas) {
		var dataURL = canvas.toDataURL().split(",")[1]
		var blobBin = atob(dataURL)

		var array = []
		for (var i = 0; i < blobBin.length; i++){
			array.push(blobBin.charCodeAt(i))
		}

		var blob = new Blob([new Uint8Array(array)], {type : 'img/jpg'})
		var formData = new FormData()
		var current_model = document.getElementById('model-selection').value 
		var model_code = current_model.split('-')[0].trim()

		formData.append('image', blob)
		formData.append('model', model_code)
		console.log(model_code)


		// send data to server
		axios({
			url : `http://${config['compute_server_ip']}:${config['compute_server_port']}/upload_and_process`,
			method : 'POST',
			data : formData,
			headers : {
				'Content-Type' : 'multipart/form-data'
			}
		}).then(response => response.data)
		.then(response => {
			this.setState({_show_map_dialog : true})
			this.setState({current_map_url : `http://${config['compute_server_ip']}:${config['compute_server_port']}/${response}`})
		})
		.catch(err => console.log(err))
	}

	/* --Util functions-- */
	exportToImage() {
		this.map.once('rendercomplete', this.getImageURL)
	  	this.map.renderSync();
	}

	getImageURL() {
		var mapCanvas = document.createElement('canvas');
	    var size = this.map.getSize();
	    mapCanvas.width = size[0];
	    mapCanvas.height = size[1];
	    var mapContext = mapCanvas.getContext('2d');
	    Array.prototype.forEach.call(
	      document.querySelectorAll('.ol-layer canvas'),
	      function (canvas) {
	        if (canvas.width > 0) {
	          var opacity = canvas.parentNode.style.opacity;
	          mapContext.globalAlpha = opacity === '' ? 1 : Number(opacity);
	          var transform = canvas.style.transform;
	          // Get the transform parameters from the style's transform matrix
	          var matrix = transform
	            .match(/^matrix\(([^]*)\)$/)[1]
	            .split(',')
	            .map(Number);
	          // Apply the transform to the export map context
	          CanvasRenderingContext2D.prototype.setTransform.apply(
	            mapContext,
	            matrix
	          );
	          mapContext.drawImage(canvas, 0, 0);
	        }
	      }
	    );
	    if (navigator.msSaveBlob) {
	      // link download attribuute does not work on MS browsers
	      navigator.msSaveBlob(mapCanvas.msToBlob(), 'map.png');
	      console.log('msSaveBlob')
	    } else {
	      var link = document.getElementById('image-download');
	      
	      // first, check if the map view is "Aerial"
	      if(this._layers[0].values_.visible) { // If Aerial is visible

		    this.sendCanvasToServer(mapCanvas)
		    // alert('Map data has been forwarded to server ... ')
	      }else{
	      	alert('Please select the plain "Aerial" map view')
	      }

	    }
	}

	showModelsDialog() {
		var export_btn = document.getElementById('export-png')
		var layer_select = document.getElementById('layer-select')
		
		if(!this.state._show_model_dialog){
			export_btn.innerHTML = 'Search tool'

			// change map layer to aerial
			layer_select.value = 'Aerial'
			for (var i = 0, ii = this._layers.length; i < ii; ++i) {
				this._layers[i].setVisible(this._styles[i] === layer_select.value);
			}
		}else{
			export_btn.innerHTML = 'Extract RoadMap'

			// change map layer to aerial with label
			layer_select.value = 'AerialWithLabelsOnDemand'
			for (var i = 0, ii = this._layers.length; i < ii; ++i) {
				this._layers[i].setVisible(this._styles[i] === layer_select.value);
			}
		}

		this.setState({_show_model_dialog : !this.state._show_model_dialog})
	}

	handleHideMapDialog() {
		this.setState({_show_map_dialog : false})
		this.setState({current_map_url : ''})
	}

	setModelsList(data){
		this.setState({models : data})
	} 

	onModelSelected() {
		var current_model = document.getElementById('model-selection').value 
		for(var i = 0; i < this.state.models.length; i++) {
			if(this.state.models[i].name === current_model){
				var latitude = this.state.models[i].latitude
				var longtitude = this.state.models[i].longtitude

				console.log(latitude, longtitude)

				var center_lonlat = [longtitude, latitude]
				var coords = fromLonLat(center_lonlat)

				this._view.setCenter(coords)
			}
		}
	}

	render() {
		return (
			<div>
				<script src="https://unpkg.com/elm-pep"></script>
				<div id="map" className="map"></div>

				<Modal show={this.state._show_map_dialog} backprop="static" keyboard={true} onHide={this.handleHideMapDialog} size="lg">
					<Modal.Header><h1>Predicted RoadMap</h1></Modal.Header>
					<Modal.Body>
						<img src={this.state.current_map_url} height={480} width={640}/>
					</Modal.Body>
				</Modal>

				<div id="utils">
					<label for="layer-select">Map Styles</label>
					<select id="layer-select" onChange={this.onChange} className="form-control">
						<option value="Aerial">Aerial</option>
						<option value="AerialWithLabelsOnDemand" selected>Aerial with labels</option>
						<option value="RoadOnDemand">Road</option>
						<option value="CanvasDark">Road dark</option>
					</select>

					<div id='coords-info'>
						<h3 className="seg-header">Coordinates info</h3><br/><br/><br/>
						<label for="center-lattitude">Map Lattitude</label>
						<input enabled={false} id='center-lattitude' value={this.state._lat} className='form-control'/>
						<label for="center-longtitude">Map longtitude</label>
						<input enabled={false} id='center-longtitude' value={this.state._long} className='form-control'/>
					</div>

					<div id='coords-utils' style={{'width':'100%'}}>
						<div id='search-by-coords-region' hidden={this.state._show_model_dialog}>
							<h3 className="seg-header">Search By Coordinates</h3><br/><br/><br/>

							<label for="countries-select">Countries</label><br/>
							<select onChange={this.onCountryChange} id="countries-select" className='form-control'>
							{this.countries.map((item, index) => {
								if(item.country === 'SG'){
									return (<option selected value={index}>{item.name}</option>)
								}else{
									return (<option value={index}>{item.name}</option>)
								}
							})}
							</select>

							<label for="search-center-lattitude">Lattitude</label>
							<input onChange={this.onLongLatChange} value={this.state._search_lat} name='_search_lat' id='search-center-lattitude' className='form-control'/>

							<label for="search-center-longtitude">longtitude</label>
							<input onChange={this.onLongLatChange} value={this.state._search_long} name='_search_long' id='search-center-longtitude' className='form-control'/>
						</div>
						<div id='extract-roadmap-region' hidden={!this.state._show_model_dialog} style={{'width':'100%'}}> 
							<h3 className='seg-header' style={{'float' :'left'}}>Choose one model</h3>
							<Modal.Dialog>
								<Modal.Body>
									<select id='model-selection'  className="form-control" onChange={this.onModelSelected}>
										<option selected="" value="0">Select model</option>
										{this.state.models.map((value, index) => {
											return (<option value={value['name']}>{value['name']}</option>)
										})}
									</select>
								</Modal.Body>
								<Modal.Footer>
									<Button onClick={this.exportToImage} style={{'width':'100%'}}>Extract and Download</Button>
								</Modal.Footer>
								<Modal.Footer>
									<p style={{'font-size':'12px'}}>Each model is associated with a particular city/country, the result might vary when a model
									is used with a city/country different from the recommended one</p>
								</Modal.Footer>
							</Modal.Dialog>
						</div>

						<table style={{'width':'100%'}}>
							<tr>
								<td style={{'width':'33.3%'}}><button style={{'width':'100%'}} onClick={this.onSearchCoords} type="button" className='btn btn-primary search-button'>Search</button></td>
								<td style={{'width':'33.3%'}}><button style={{'width':'100%'}} onClick={this.onDefaultCoords} type="button" className='btn btn-primary search-button'>Default</button></td>
								<td style={{'width':'33.3%'}}><button style={{'width':'100%'}} id="export-png" class="btn btn-primary" onClick={this.showModelsDialog}><i class="fa fa-download"></i>Extract RoadMap</button></td>
    						</tr>	
    					</table>
    					<a id="image-download" download="map.png"></a>
					</div>
				 </div>
			</div>
		)
	}
}

export default MapView

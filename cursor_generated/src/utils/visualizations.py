"""
Visualization utilities for the Aura Shipping Intelligence Platform.

This module provides functions for creating charts and visualizations
for shipping data analysis.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional


def create_pricing_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a pricing analysis chart from Fedex pricing data.
    
    Args:
        df: DataFrame with columns: weight, price, service_type, zone
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        # Return empty chart if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No pricing data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create scatter plot of weight vs price
    fig = px.scatter(
        df, 
        x="weight", 
        y="price",
        color="service_type",
        size="zone",
        hover_data=["zone", "transportation_type"],
        title="Fedex Pricing Analysis",
        labels={
            "weight": "Package Weight (lbs)",
            "price": "Price (USD)",
            "service_type": "Service Type",
            "zone": "Zone"
        }
    )
    
    fig.update_layout(
        xaxis_title="Package Weight (lbs)",
        yaxis_title="Price (USD)",
        title_x=0.5
    )
    
    return fig


def create_zone_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a zone distance mapping chart.
    
    Args:
        df: DataFrame with columns: zone, min_distance, max_distance
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        # Return empty chart if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No zone data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create bar chart showing distance ranges for each zone
    fig = go.Figure()
    
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=f"Zone {row['zone']}",
            x=[f"Zone {row['zone']}"],
            y=[row['max_distance'] - row['min_distance']],
            base=row['min_distance'],
            text=f"{row['min_distance']}-{row['max_distance']} miles",
            textposition='auto',
            hovertemplate="Zone %{x}<br>Distance: %{text}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Zone Distance Mapping",
        xaxis_title="Zone",
        yaxis_title="Distance Range (miles)",
        barmode='group',
        title_x=0.5
    )
    
    return fig


def create_weather_chart(weather_data: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create a weather visualization chart.
    
    Args:
        weather_data: Dictionary containing weather information
        
    Returns:
        Plotly figure object or None if no data
    """
    if not weather_data:
        return None
    
    # Extract temperature data for different locations
    locations = list(weather_data.keys())
    temperatures = []
    
    for location in locations:
        current = weather_data[location].get("current", {})
        temp = current.get("temperature_2m", 0)
        temperatures.append(temp)
    
    fig = go.Figure(data=[
        go.Bar(
            x=locations,
            y=temperatures,
            text=[f"{temp}°F" for temp in temperatures],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Current Temperature by Location",
        xaxis_title="Location",
        yaxis_title="Temperature (°F)",
        title_x=0.5
    )
    
    return fig


def create_fuel_price_chart(fuel_data: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create a fuel price visualization chart.
    
    Args:
        fuel_data: Dictionary containing fuel price information
        
    Returns:
        Plotly figure object or None if no data
    """
    if not fuel_data:
        return None
    
    fuel_types = list(fuel_data.keys())
    prices = [fuel_data[ft].get("current_price", 0) for ft in fuel_types]
    
    fig = go.Figure(data=[
        go.Bar(
            x=fuel_types,
            y=prices,
            text=[f"${price:.2f}" for price in prices],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Current Fuel Prices",
        xaxis_title="Fuel Type",
        yaxis_title="Price (USD/gallon)",
        title_x=0.5
    )
    
    return fig


def create_traffic_chart(traffic_data: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create a traffic visualization chart.
    
    Args:
        traffic_data: Dictionary containing traffic information
        
    Returns:
        Plotly figure object or None if no data
    """
    if not traffic_data:
        return None
    
    locations = list(traffic_data.keys())
    speeds = [traffic_data[loc].get("average_speed", 0) for loc in locations]
    
    fig = go.Figure(data=[
        go.Bar(
            x=locations,
            y=speeds,
            text=[f"{speed} mph" for speed in speeds],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Average Traffic Speed by Location",
        xaxis_title="Location",
        yaxis_title="Average Speed (mph)",
        title_x=0.5
    )
    
    return fig 